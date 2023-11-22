kill -9 $(ps aux | grep 'python' | grep 'fastchat' | grep -v 'grep' | awk '{print $2}')
