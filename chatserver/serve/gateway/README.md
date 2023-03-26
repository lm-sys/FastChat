# ChatServer Nginx Gateway

## Purpose of the Gateway

The Nginx gateway serves the following purposes:

1. Protects Gradio servers by acting as a firewall.
2. Facilitates dynamic mounting and unmounting of Gradio servers.
3. Provides load balancing for Gradio servers.
4. Offers additional security features, such as total connection limit.
5. Reduces attack surface by requiring only a single public port to be exposed for serving.

## Deployment and Updating of the Gateway

### Installing Nginx

On Debian-based distributions (e.g., Ubuntu):

```bash
sudo apt update
sudo apt install nginx
```
On Red Hat-based distributions (e.g., CentOS, Fedora):

```bash
sudo yum install epel-release
sudo yum install nginx
```

### Depolyment

Copy `nginx.conf` to `/etc/nginx/nginx.conf` (need sudo permission).

Replace the port number 7960 in `listen 7960` with the desired port for the gateway.

Modify `upstream websocket` to config Gradio servers behind the gateway.

Lastly, update Nginx.

### Updating

When `/etc/nginx/nginx.conf` is modified, you need to update the Nginx service:

```bash
sudo nginx -t  # check `/etc/nginx/nginx.conf`
sudo systemctl reload nginx  # restart Nginx service to load the new config
sudo systemctl status nginx  # check the status of the Nginx service. It should be active (running).
```

