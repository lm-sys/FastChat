kill -9 $(ps aux | grep 'python' | grep -v 'grep' | awk '{print $2}')
