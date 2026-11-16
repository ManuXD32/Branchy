# Setting up workers and a coordinator

One machine is the **coordinator**: it holds the models, decides which device computes what, and runs the web UI
and the benchmarks. Every other device is a **worker**: a small Python daemon (the "node") that the coordinator
controls over TCP port 8730. Workers are installed with one command and then updated and configured from the
coordinator; you never log into them again.

Tested: Linux x86-64 (coordinator and worker) and Android phones with Termux (worker, and coordinator through a
job). All devices must reach each other over IP (LAN or a VPN such as WireGuard or Tailscale).

## 1. Coordinator

```bash
git clone <this repository> && cd <this repository>
python3 -m venv .venv && . .venv/bin/activate          # any Python 3.10+ environment works
pip install -r requirements.txt
BASE=<where the model files are published> bash scripts/get_models.sh      # see MODELS.md
bash scripts/setup_coordinator.sh 192.168.1.10          # the address of this machine as your devices see it
```

`setup_coordinator.sh` creates a shared secret (`.secrets/node_token`, never commit it), builds the worker bundle
into `dist/` and creates `src/node/cluster.json` from the example. It then prints the next commands.

## 2. Workers

Serve the bundle from the coordinator while you install (stop it afterwards):

```bash
python3 -m http.server 8765 --bind 192.168.1.10 --directory dist
```

On each device, in a shell (on Android: install Termux from F-Droid, open it):

```bash
curl -fsSL http://192.168.1.10:8765/go.sh | BRANCHY_TOKEN=<content of .secrets/node_token> BRANCHY_NAME=phone bash
```

The installer unpacks into `~/branchy-node`, finds a Python with torch (on Termux it installs
`python python-numpy python-torch` with `pkg`; on Linux it makes a private venv with CPU torch if needed), and
starts the node inside a tmux session called `branchy`. On Linux with a firewall, open the port first:
`sudo firewall-cmd --add-port=8730/tcp`. Re-running the command is safe and is the recovery path if a node dies.
On Android, keep Termux allowed to run in the background (the installer takes the wake lock).

**Security:** a node runs code that the coordinator sends, with the rights of the user who started it, for anyone
who knows the secret. Use it only on networks you control, keep the secret private, and stop nodes you do not need
(`bash ~/branchy-node/stop.sh`).

## 3. Cluster file

List your workers in `src/node/cluster.json` (names are yours to choose):

```json
{"token_file": ".secrets/node_token",
 "nodes": {"laptop": {"host": "192.168.1.20", "port": 8730},
           "phone":  {"host": "192.168.1.30", "port": 8730}}}
```

Check and calibrate from `src/node/`:

```bash
python nodectl.py info        # who answers, versions, threads
python nodectl.py selftest    # every node must compute the same numbers as this machine
python nodectl.py ping        # round-trip times
python nodectl.py profile     # speed of a synthetic branch on every node
python nodectl.py restart phone --threads 4 --kernels math-attn      # change a node's settings
python nodectl.py push        # send the current worker code to every node (after you change src/node/payload)
```

Settings that mattered for us: Android phones need `--kernels math-attn` (the built-in attention of the Termux
torch build gave wrong numbers with several threads) and ran best with 4 threads; on torch builds that compute with
OpenBLAS use `--linear mv` and set `OPENBLAS_NUM_THREADS` to the torch thread count on the coordinator.

## 4. Web UI

```bash
bash scripts/start_webui.sh --threads 8            # http://127.0.0.1:8780
```

- **Cluster**: which workers answer, their settings and round-trip time.
- **Compare**: several configurations write from the same beginning, one after the other: the sequential model
  on this machine, its layer split over workers, the branched model with branches on workers. You see the text,
  tokens per second and where the time of a token goes.
- **Benchmark**: time a list of placements; every placement must reproduce the output of the first one.

The page has no login. It listens on 127.0.0.1 unless you pass `--bind`.

## 5. A worker as coordinator

Any worker can coordinate (useful when the machine you sit at is slow or far away):

```bash
cd src/node
python remote_coord.py --coordinator laptop --peers phone --checkpoint ../../models/branched8-309M/inference.pt \
  --placement "alone=local:0,1,2,3,4,5,6,7" --placement "4+4=local:0,1,2,3;phone:4,5,6,7" --out ../../results/laptop_coordinates
```
