# NtfySummarizer

NtfySummarizer is a tool that uses [ntfy.sh](ntfy.sh) and [wdoc](https://github.com/thiswillbeyourgithub/wdoc) to receive URLs, generate summaries of web content, and send the summaries back via ntfy.sh notifications. You can use it for example to get the summary of a webpage or youtube videos directly on your phone.

## Overview

This tool allows you to:
1. Receive URLs through an ntfy.sh topic
2. Process the URLs to generate summaries using wdoc
3. Send the summaries back through another ntfy.sh topic

## Prerequisites
- [ntfy.sh](https://ntfy.sh) installed on your phone and computer
- [wdoc](https://github.com/thiswillbeyourgithub/wdoc/)

## Setup

1. Install the required Python packages:
   ```
   pip install requests fire beartype
   ```

2. Clone this repository or download the files to your local machine.

3. Rename `TOPICS.EXAMPLE` to `TOPICS` and set your desired ntfy.sh topics:
   ```
   cp TOPICS.EXAMPLE TOPICS
   ```
   Edit `TOPICS` to set your preferred topic names for receiving URLs and sending summaries.

## How to Use

1. Start the NtfySummarizer service:
   ```
   eval $(cat TOPICS) ; ntfy subscribe $topic_receive $(pwd)/script.sh
   ```
   This command sets up the environment variables and starts listening for incoming URLs on the specified topic.

Alternatively, you can use the systemctl by putting this in /etc/ntfy/client.yml:
```
subscribe:
  - topic: [your 'topics_receive']
    command: /PATH/TO/HERE/script.sh
```
Then `sudo systemctl restart ntfy-client.service`.

2. Send a URL to summarize:
   Use ntfy.sh to send a message to the `$topic_receive` topic. The message should be either:
   - A URL starting with "http" or "https"
   - A file type followed by a URL, e.g., "pdf https://example.com/document.pdf"

3. Receive summaries:
   The tool will process the URL, generate a summary, and send it to the `$topic_send` topic. You'll receive a notification with the summary on any device subscribed to this topic.

## Notes

- If the URL doesn't start with "http", the format `[filetype] [url]` is assumed.
- The summary will include the original URL, a markdown-formatted summary, the total cost of generating the summary, and the estimated reading time saved.
- Ntfy.sh ensures that if a message is more than 4096 bytes long it will be sent as a file attachment. We do that automatically here to make sure the file sent is as .md and not .txt
- You can pass the argument `render_md=True` to pass the text through [rich markdown rendering](https://github.com/Textualize/rich) if you prefer it (Ntfy on android is not able to render markdown).

## Troubleshooting

- Ensure that the `TOPICS` file is properly configured with your desired topic names.
- Check that you have the necessary permissions to execute `script.sh`. In my case I had to modify `/lib/systemd/system/ntfy-client.service` to change `User=` and `Group=` to my own username instead of `ntfy`, otherwise I had '`Permission denied`.
- Verify that the wdoc library is correctly installed and configured in your Python environment.

For more information on ntfy.sh usage, visit the [ntfy.sh documentation](https://ntfy.sh/docs/).
