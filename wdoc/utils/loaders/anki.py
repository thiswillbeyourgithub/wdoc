import json
import re
import shutil
import warnings
from pathlib import Path

import bs4
import uuid6
from beartype.typing import Dict, List, Optional, Tuple, Union
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from loguru import logger
from tqdm import tqdm

from wdoc.utils.env import env, is_out_piped
from wdoc.utils.loaders.shared import debug_return_empty
from wdoc.utils.misc import file_hasher, html_to_text, optional_strip_unexp_args

clozeregex = re.compile(r"{{c\d+::|}}")  # for removing clozes in anki
anki_replacements_regex = re.compile(r"\{([^}]*)\}")

REG_IMG = re.compile(r"<img .*?src=.*?/?>", flags=re.MULTILINE | re.DOTALL)

REG_SOUNDS = re.compile(
    r"\[sound:\w+\.\w{2,3}\]",
)
REG_LINKS = re.compile(
    r"[A-Za-z0-9]+://[A-Za-z0-9%-_]+(?:/[A-Za-z0-9%-_])*(?:#|\\?)[A-Za-z0-9%-_&=]*",
)

STR_IMAGE_OCR = "{image_ocr_alt}"


def cloze_stripper(clozed: str) -> str:
    clozed = clozeregex.sub(" ", clozed)
    return clozed


@debug_return_empty
@optional_strip_unexp_args
def load_anki(
    verbose: bool,
    text_splitter: TextSplitter,
    loaders_temp_dir: Path,
    anki_profile: Optional[str] = None,
    anki_deck: Optional[str] = None,
    anki_notetype: Optional[str] = None,
    anki_template: Optional[str] = "{allfields}\n" + STR_IMAGE_OCR,
    anki_tag_filter: Optional[str] = None,
    anki_tag_render_filter: Optional[str] = None,
) -> List[Document]:
    import ankipandas as akp

    if anki_tag_render_filter:
        assert (
            "{tags}" in anki_template
        ), "Can't use anki_tag_render_filter without using {tags} in anki_template"
        try:
            anki_tag_render_filter = re.compile(anki_tag_render_filter)
        except Exception as err:
            raise Exception(f"Failed to compile anki_tag_render_filter: '{err}'")

    if anki_tag_filter:
        try:
            anki_tag_filter = re.compile(anki_tag_filter)
        except Exception as err:
            raise Exception(f"Failed to compile anki_tag_filter: '{err}'")

    if not anki_profile:
        original_db = akp.find_db()
        anki_profile = original_db.parent.name
        logger.info(f"Detected anki profile: '{anki_profile}'")

    logger.info(f"Loading anki profile: '{anki_profile}'")
    original_db = akp.find_db(user=anki_profile)
    name = f"{anki_profile}".replace(" ", "_")
    random_val = str(uuid6.uuid6())
    new_db_path = (
        loaders_temp_dir / f"anki_collection_{name.replace('/', '_')}_{random_val}"
    )
    assert not Path(new_db_path).exists(), f"{new_db_path} already existing!"
    shutil.copy(original_db, new_db_path)
    col = akp.Collection(path=new_db_path)
    cards = col.cards.merge_notes()

    if verbose and not is_out_piped:
        tqdm.pandas()

        def pbar(*x, **y):
            tqdm.pandas(*x, **y)

    else:
        import pandas as pd

        pd.DataFrame.progress_apply = pd.DataFrame.apply
        pd.Series.progress_apply = pd.Series.apply

        def pbar(*x, **y):
            pass

    cards.loc[cards["codeck"] == "", "codeck"] = cards["cdeck"][cards["codeck"] == ""]

    cards["codeck"] = cards["codeck"].progress_apply(lambda x: x.replace("\x1f", "::"))
    if anki_deck:
        cards = cards[cards["codeck"].str.startswith(anki_deck)]
    cards["nmodel"] = cards["nmodel"].progress_apply(lambda x: x.lower())
    if anki_notetype:
        cards = cards[cards["nmodel"].str.contains(anki_notetype, case=False)]
        assert (
            not cards.empty
        ), f"No cards found after filtering by notetype {anki_notetype}"
    if anki_tag_filter:
        pbar(desc="Filtering by tags")
        cards = cards[
            cards.progress_apply(
                (lambda x: any(anki_tag_filter.match(t) for t in x["ntags"])), axis=1
            )
        ]
        assert (
            not cards.empty
        ), f"No cards found after filtering by tags: {anki_tag_filter}"

    # remove suspended
    cards = cards[cards["cqueue"] != "suspended"]

    # merge models and fields for easy handling
    cards["mid"] = col.cards.mid.loc[cards.index]
    mid2fields = akp.raw.get_mid2fields(col.db)
    # make the model fields lowercase
    mid2fields = {
        k: (lambda x: [y.lower() for y in x])(v) for k, v in mid2fields.items()
    }
    # mod2mid = akp.raw.get_model2mid(col.db)
    cards["fields_name"] = cards["mid"].progress_apply(lambda x: mid2fields[x])
    assert not cards.empty, "empty dataframe!"

    # remove duplicate, essentially making cards the same thing as notes
    cards = cards.drop_duplicates(subset="nid", keep="first")
    notes = cards.reset_index().set_index("nid")

    # check placeholders validity
    placeholders = [ph.lower() for ph in anki_replacements_regex.findall(anki_template)]
    assert placeholders, f"No placeholder found in anki_template '{anki_template}'"
    for ph in placeholders:
        for ic, c in notes.iterrows():
            if ph not in c["fields_name"] + ["allfields", "tags", STR_IMAGE_OCR[1:-1]]:
                raise Exception(
                    "A placeholder in anki template didn't match fields of "
                    f"a card.\nCulprit placeholder: {ph}\nTemplate: "
                    f"{anki_template}\nExample card: {c}"
                )

    # prepare field values
    if "{allfields}" in anki_template:
        useallfields = True
        pbar(desc="Parsing allfields value")
        notes["allfields"] = notes.progress_apply(
            lambda x: "\n\n".join(
                [
                    f"{k.lower()}: '{html_to_text(cloze_stripper(v)).strip()}'"
                    for k, v in zip(x["fields_name"], x["nflds"])
                    if v.strip()
                ]
            ),
            axis=1,
        )
    else:
        useallfields = False

    if STR_IMAGE_OCR in anki_template:
        useimageocr = True
    else:
        useimageocr = False

    if "{tags}" in anki_template:
        usetags = True
        pbar(desc="Formatting tags")
        notes["tags_formatted"] = notes.progress_apply(
            lambda x: (
                (
                    "\n"
                    + "\n".join(
                        [
                            t
                            for t in x["ntags"]
                            if (
                                anki_tag_render_filter is None
                                or anki_tag_render_filter.match(t)
                            )
                        ]
                    ).strip()
                    + "\n"
                )
                if x["ntags"]
                else ""
            ),
            axis=1,
        )
        if notes["ntags"].notnull().any():
            assert (
                notes["tags_formatted"].notnull().any()
            ), "No tags were extracted because of your filter. Crashing to let you recheck your setup."
    else:
        usetags = False

    def placeholder_replacer(row: "pd.Series") -> Tuple[str, dict]:
        text = anki_template

        if useallfields:
            text = text.replace("{allfields}", row["allfields"])
        if usetags:
            text = text.replace("{tags}", row["tags_formatted"])

        for ph in placeholders:
            if ph == "tags" or ph == "allfields" or ph == STR_IMAGE_OCR[1:-1]:
                continue
            field_val = row["nflds"][row["fields_name"].index(ph)]
            text = text.replace(
                "{" + ph + "}",
                html_to_text(
                    cloze_stripper(field_val),
                ),
            )
        text = text.replace("\\n", "\n").replace("\\xa0", " ")

        # replace media
        new_text, medias = replace_media(
            content=text,
            media=None,
            mode="remove_media",
            strict=False,
            replace_links=False,
        )
        if medias:
            assert text != new_text
        text = new_text
        if useimageocr:
            image_keys = [k for k in medias.keys() if "IMAGE" in k]
            for img_k in image_keys:
                img = bs4.BeautifulSoup(medias[img_k], "html.parser")
                title = img.get("title").strip() if img.has_attr("title") else ""
                alt = img.get("alt").strip() if img.has_attr("alt") else ""
                ocr_alt = ""
                if title:
                    ocr_alt += f"\nTitle: '{title}'"
                if alt:
                    ocr_alt += f"\nAlt: '{alt}'"
                ocr_alt = ocr_alt.strip()
                if ocr_alt:
                    text = text.replace(
                        STR_IMAGE_OCR,
                        f"\n<OCR of '{k}'>\n{ocr_alt}\n</OCR of '{k}'>" + STR_IMAGE_OCR,
                    )
            text = text.replace(STR_IMAGE_OCR, "").strip()

        return text, medias

    pbar(desc="Formatting all cards")
    notes["medias"] = {}
    out = notes.progress_apply(placeholder_replacer, axis=1)
    notes["text"] = [t[0] for t in out]
    notes["medias"] = [t[1] for t in out]

    notes["text"] = notes["text"].progress_apply(lambda x: x.strip())
    notes = notes[notes["text"].ne("")]  # remove empty text

    # remove notes that contain an image, sound or link
    # notes = notes[~notes["text"].str.contains("\[IMAGE_")]
    # notes = notes[~notes["text"].str.contains("\[SOUND_")]
    # notes = notes[~notes["text"].str.contains("\[LINK_")]

    notes["text"] = notes["text"].apply(lambda x: x.strip())
    notes = notes[notes["text"].ne("")]  # remove empty text
    notes.drop_duplicates(subset="text", inplace=True)

    notes = notes.sort_index()

    docs = []

    # load each card as a single document
    for nid, c in notes.iterrows():
        assert c["codeck"], f"empty card_deck for nid {nid}"
        # turn the media into absolute paths
        medias = c["medias"]
        to_add = {}
        for k, v in medias.items():
            assert (
                k in c["text"]
            ), f"missing media '{k}' in text '{c['text']}' of card '{c}'"
            try:
                src = bs4.BeautifulSoup(v, "html.parser").find("img")["src"]
                assert src
                v = Path(original_db).parent / "collection.media" / src
                v = v.resolve()
                if v.exists():
                    if k in c["text"]:
                        h = file_hasher({"path": str(v.absolute())})[:6]
                        placeholder = f"IMAGE_{h}"
                        medias[k] = None
                        to_add[placeholder] = str(v.absolute())
                        c["text"] = c["text"].replace(k, placeholder)
                    else:
                        medias[k] = str(v.absolute())
            except Exception:
                # it was probably not a file
                continue
        medias = {k: v for k, v in medias.items() if v is not None}
        if to_add:
            medias.update(to_add)
            assert all(k in c["text"] for k in to_add.keys())
        # better formatting for tags
        ntags = [
            nt
            # bettter for the tokenizer I guess
            # nt.replace("_", " ").replace("-", " ").replace("::", " > ")
            for nt in c["ntags"]
        ]
        docs.append(
            Document(
                page_content=c["text"],
                metadata={
                    "anki_tags": " ".join(ntags),
                    "anki_nid": str(nid),
                    "anki_deck": c["codeck"],
                    "anki_modtime": int(c["cmod"]),
                    "anki_media": json.dumps(medias, ensure_ascii=False),
                },
            )
        )

    assert docs, "List of loaded anki document is empty!"

    path = (
        f"Anki_profile='{anki_profile}',deck='{anki_deck}',notetype='{anki_notetype}'"
    )
    for i in range(len(docs)):
        docs[i].metadata["anki_profile"] = anki_profile
        docs[i].metadata["anki_topdeck"] = anki_deck
        docs[i].metadata["anki_notetype"] = anki_notetype
        docs[i].metadata["path"] = path
        docs[i].metadata["anki_nid"] = " ".join(
            sorted(docs[i].metadata["anki_nid"].split(" "))
        )

    # delete temporary db file
    new_db_path.unlink()
    Path(str(new_db_path.absolute()) + "-shm").unlink(missing_ok=True)
    Path(str(new_db_path.absolute()) + "-wal").unlink(missing_ok=True)
    return docs


def replace_media(
    content: str,
    media: Union[None, Dict],
    mode: str,
    strict: bool = True,
    replace_image: bool = True,
    replace_links: bool = True,
    replace_sounds: bool = True,
) -> Tuple[str, Dict]:
    """
    Else: exclude any note that contains in the content:
        * an image (<img...)
        * or a sound [sound:...
        * or a link href / http
    This is because:
        1 as LLMs are non deterministic I preferred
            to avoid taking the risk of botching the content
        2 it costs less token

    The intended use is to call it first to replace
    each media by a simple string like [IMAGE_1] and check if it's
    indeed present in the output of the LLM then replace it back.

    It uses both bs4 and regex to be sure of itself
    """
    # ignore warnings from beautiful soup that can happen because anki is not exactly html
    warnings.filterwarnings("ignore", category=UserWarning, module="bs4")

    assert mode in ["add_media", "remove_media"]
    assert content.strip()
    if media is None:
        media = {}
    assert isinstance(media, dict)
    assert any(rule for rule in [replace_sounds, replace_links, replace_image])

    if mode == "remove_media":
        assert not media
        images = []
        sounds = []
        links = []

        if replace_links:
            # fix links common issues
            content = content.replace(":// ", "://")
            content = content.replace("http ://", "http://")
            content = content.replace("https ://", "http://")

        # Images
        if replace_image and "<img" in content:
            soup = bs4.BeautifulSoup(content, "html.parser")
            images_bs4 = [str(img) for img in soup.find_all("img")]
            # fix bs4 parsing as ending with /> instead of >
            images_bs4 = [
                (
                    img[:-2] + ">"
                    if ((img not in content) and img[:-2] + ">" in content)
                    else img
                )
                for img in images_bs4
            ]
            images_reg = re.findall(REG_IMG, content)
            if len(images_bs4) != len(images_reg):
                if env.WDOC_VERBOSE:
                    logger.warning(
                        f"Different images found:\nbs4: {images_bs4}\nregex: {images_reg}\nContent: {content}"
                    )
                if images_bs4 and not images_reg:
                    images = [str(img) for img in images_bs4]
                elif (not images_bs4) and images_reg:
                    images = [str(img) for img in images_reg]
            else:
                images = [str(img) for img in images_bs4]
            try:
                assert images, f"no image found but should have. Text is '{content}'"
            except AssertionError as err:
                if strict:
                    raise
                logger.warning(err)
            for iimg, img in enumerate(images):
                try:
                    assert (
                        img in content
                    ), f"missing img from content:\nimg: {img}\ncontent: {content}"
                    assert re.search(
                        REG_IMG, img
                    ), f"Regex couldn't identify img: {img}"
                    assert not re.search(
                        REG_SOUNDS, img
                    ), f"Sound regex identifier img: {img}"
                except AssertionError as err:
                    if strict:
                        raise
                    logger.warning(err)
                    images[iimg] = None
            images = [i for i in images if i is not None]
            images = list(set(images))

        # Sounds
        if replace_sounds and "[sounds:" in content:
            sounds = re.findall(REG_SOUNDS, content)
            try:
                assert sounds, f"No sounds found but should have. Content: {content}"
            except AssertionError as err:
                if strict:
                    raise
                logger.warning(err)
            for isound, sound in enumerate(sounds):
                try:
                    assert sound in content, f"Sound is not in content: {sound}"
                    assert not re.search(
                        REG_IMG, sound
                    ), f"Image regex identified this sound: {sound}"
                    assert re.search(
                        REG_SOUNDS, sound
                    ), f"Regex didn't identify this sound: {sound}"
                except AssertionError as err:
                    if strict:
                        raise
                    logger.warning(err)
                    sounds[isound] = None
            sounds = [s for s in sounds if s is not None]
            sounds = list(set(sounds))

        # links
        if replace_links and "://" in content:
            links = re.findall(REG_LINKS, content)
            links = [
                link
                for link in links
                if not any(other != link and other in link for other in links)
            ]
            if strict:
                assert links, "No links found"
            for ilink, link in enumerate(links):
                try:
                    assert (
                        link in content
                    ), f"Link not in content:\nlink: {link}\ncontent: {content}"
                    assert re.search(
                        REG_LINKS, link
                    ), f"Regex couldn't identify link: {link}"
                except AssertionError as err:
                    if strict:
                        raise
                    logger.warning(err)
                    links[ilink] = None
            links = [li for li in links if li is not None]
            links = list(set(links))

        if not images + sounds + links:
            return content, {}

        new_content = content

        # do the replacing
        for i, img in enumerate(images):
            assert replace_image, replace_image
            try:
                assert img in content, f"img '{img}' not in content '{content}'"
                assert (
                    img in new_content
                ), f"img '{img}' not in new_content '{new_content}'"
                assert img not in media.keys() and img not in media.values()
                replaced = f"[IMAGE_{i+1}]"
                assert replaced not in media.keys() and replaced not in media.values()
                assert (
                    replaced not in content
                ), f"Replaced '{replaced}' already in content '{content}'"
                assert (
                    replaced not in new_content
                ), f"Replaced '{replaced}' already in new_content '{new_content}'"
                new_content = new_content.replace(img, replaced)
                media[replaced] = img
                assert img not in new_content
                assert replaced in new_content
            except AssertionError as err:
                if strict:
                    raise
                logger.warning(f"Failed assert when replacing image: '{err}'")
                continue

        for i, sound in enumerate(sounds):
            try:
                assert replace_sounds
                assert sound in content
                assert sound in new_content
                assert sound not in media.keys() and sound not in media.values()
                replaced = f"[SOUND_{i+1}]"
                assert replaced not in media.keys() and replaced not in media.values()
                assert replaced not in content
                assert replaced not in new_content
                new_content = new_content.replace(sound, replaced)
                media[replaced] = sound
                assert sound not in new_content
                assert replaced in new_content
            except AssertionError as err:
                if strict:
                    raise
                logger.warning(f"Failed assert when replacing sounds: '{err}'")
                continue

        for i, link in enumerate(links):
            try:
                assert replace_links
                assert link in content
                assert link not in media.keys()
                replaced = f"[LINK_{i+1}]"
                assert replaced not in media.keys() and replaced not in media.values()
                assert replaced not in content
                assert replaced not in new_content
                assert link in new_content or len(
                    [val for val in media.values() if link in val]
                )
                if link not in new_content:
                    continue
                else:
                    new_content = new_content.replace(link, replaced)
                    media[replaced] = link
                    assert link not in new_content
                    assert replaced in new_content
            except AssertionError as err:
                if strict:
                    raise
                logger.warning(f"Failed assert when replacing links: '{err}'")
                continue

        # check no media can be found anymore
        if replace_image:
            if strict:
                assert not re.findall(REG_IMG, new_content), new_content
                assert not bs4.BeautifulSoup(new_content, "html.parser").find_all(
                    "img"
                ), new_content
                assert "<img" not in new_content, new_content
            elif "<img" in new_content:
                logger.warning(f"AnkiMediaReplacer: Found '<img' in '{new_content}'")
        if replace_sounds:
            if strict:
                assert not re.findall(REG_SOUNDS, new_content), new_content
                assert "[sound:" not in new_content, new_content
            elif "[sound:" in new_content:
                logger.warning(f"AnkiMediaReplacer: Found '[sound:' in '{new_content}'")
        if replace_links:
            if strict:
                assert not re.findall(REG_LINKS, new_content), new_content
                assert "://" not in new_content, new_content
            elif "://" in new_content:
                logger.warning(f"AnkiMediaReplacer: Found '://' in '{new_content}'")

        # check non empty
        temp = new_content
        for med, val in media.items():
            temp = temp.replace(med, "")
        assert temp.strip()

        # recursive check:
        assert (
            replace_media(
                content=new_content,
                media=media,
                mode="add_media",
                strict=strict,
                replace_image=replace_image,
                replace_links=replace_links,
                replace_sounds=replace_sounds,
            )[0]
            == content
        )

        return new_content, media

    elif mode == "add_media":
        assert media

        # TODO check that all media are found
        new_content = content
        for med, val in media.items():
            assert med in content
            assert val not in content
            assert val not in new_content
            new_content = new_content.replace(med, val)
            assert med not in new_content
            assert val in new_content

        return new_content, {}

    else:
        raise ValueError(mode)
