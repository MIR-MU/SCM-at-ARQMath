#!/bin/sh
# Retrieves a Math stackexchange question ID and title for an answer ID.
curl -s --location https://math.stackexchange.com/questions/$1 |
  sed -n -r '/class="question-hyperlink"/s#.*<a href="/questions/([0-9]*)/.*class="question-hyperlink">(.*)</a>.*#\1 - \2#p' | head -1
