#!/bin/zsh

local_path="${0:h}"

# Check if TOPICS file exists
if [[ ! -f "$local_path/TOPICS" ]]; then
    echo "Error: TOPICS file not found in $local_path"
    exit 1
fi

# # Source shell configuration files if they exist, this can be useful for some user that for example declared specific env variables
for rcfile in ~/.zshrc ~/.bashrc ~/.bash_profile; do
    if [[ -f "$rcfile" ]]; then
        source "$rcfile"
        break
    fi
done

# cd into wdoc, useful to trigger some autoenv things for some users
cd $local_path/../../

eval $(cat $local_path/TOPICS)

# Check if required variables are set
if [[ -z "$topic_receive" || -z "$topic_send" ]]; then
    echo "Error: TOPICS file must declare both topic_receive and topic_send variables"
    exit 1
fi

output=$(python "${0:h}"/NtfySummarizer.py --topic="$topic_send" 2>&1 >/dev/null) && echo "Success" && exit 0

mess="An error happened during wdoc execution on input \"$NTFY_MESSAGE\"

Full output:
$output"

# the topic has to be declared twice otherwise the env has priority
NTFY_TOPIC="$topic_send" ntfy publish $topic_send "$mess" || echo $mess
