# Test-only isolated agent host for the Docker simulation.
#
# It is a pre-provisioned ssh host: sshd, Python 3.12 (for hmz's remote anchor) and the
# fake ACP agent.  No provider key, no repository and no control-host home are copied in.
FROM ubuntu:24.04

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates openssh-server python3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -s /bin/bash agent \
    && mkdir -p /srv/taic /run/sshd /home/agent/.ssh \
    && chown agent:agent /srv/taic /home/agent/.ssh \
    && chmod 700 /home/agent/.ssh

COPY fake_acp.py /opt/taic/fake_acp.py
RUN chmod 644 /opt/taic/fake_acp.py

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D", "-e"]
