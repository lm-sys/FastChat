kill -9 $(ps aux | grep 'python3' | grep -v 'grep' | awk '{print $2}')
