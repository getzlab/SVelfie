# Use cicirello/pyaction:4.14.0 as base image for docker 
FROM cicirello/pyaction:4.14.0

# Install/set up from root directory
WORKDIR /

### INSTALLATIONS:
# Clone SSVGAR (original name for SVelfie) Github repo
RUN git clone https://ssvgar-dummy-user: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@github.com/getzlab/SVelfie.git
# Install requirements.txt for this version of SSVGAR
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r SVelfie/run/requirements.txt
# Install bedtools 2.27.1
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null && \
    apt update && \ 
    apt-get install -y wget && apt-get install -y build-essential && apt-get install -y libz-dev
RUN wget https://github.com/arq5x/bedtools2/releases/download/v2.27.1/bedtools-2.27.1.tar.gz && \
    tar -zxvf bedtools-2.27.1.tar.gz
WORKDIR /bedtools2
RUN make
ENV PATH="/bedtools2/bin:$PATH"
# Install text editors for convenience (vim and emacs)
RUN apt install -y vim && apt install -y emacs

# Set the working directory
WORKDIR /SVelfie
