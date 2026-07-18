"""Dispatch a list of shell commands across GPUs with per-GPU concurrency.

Usage: python launch.py --jobs jobs.txt --gpus 2,3,4,5,6,7 --per_gpu 3
Each line of jobs.txt is one command; CUDA_VISIBLE_DEVICES is prepended.
Failed jobs are reported at the end (and to failed_jobs.txt).
"""
import argparse, subprocess, sys, threading, queue, time, os


def worker(q, gpu, results, log_dir):
    while True:
        try:
            idx, cmd = q.get_nowait()
        except queue.Empty:
            return
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        log = os.path.join(log_dir, f"job{idx:03d}.log")
        with open(log, 'w') as f:
            f.write(cmd + "\n\n")
            f.flush()
            t0 = time.time()
            r = subprocess.run(cmd, shell=True, env=env, stdout=f,
                               stderr=subprocess.STDOUT)
        results[idx] = (r.returncode, round(time.time() - t0), cmd)
        status = 'OK ' if r.returncode == 0 else 'FAIL'
        print(f"[{status}] gpu{gpu} job{idx:03d} {round(time.time()-t0)}s :: "
              f"{cmd[:120]}", flush=True)
        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--gpus', default='2,3,4,5,6,7')
    ap.add_argument('--per_gpu', type=int, default=3)
    ap.add_argument('--log_dir', required=True)
    args = ap.parse_args()

    cmds = [l.strip() for l in open(args.jobs) if l.strip() and not l.startswith('#')]
    os.makedirs(args.log_dir, exist_ok=True)
    q = queue.Queue()
    for i, c in enumerate(cmds):
        q.put((i, c))
    results = {}
    threads = []
    for gpu in args.gpus.split(','):
        for _ in range(args.per_gpu):
            t = threading.Thread(target=worker, args=(q, gpu, results, args.log_dir),
                                 daemon=True)
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    fails = [(i, c) for i, (rc, _, c) in results.items() if rc != 0]
    print(f"\n{len(cmds) - len(fails)}/{len(cmds)} jobs succeeded")
    if fails:
        with open(os.path.join(args.log_dir, 'failed_jobs.txt'), 'w') as f:
            for i, c in fails:
                f.write(c + "\n")
                print(f"FAILED job{i:03d}: {c[:140]}")
        sys.exit(1)


if __name__ == '__main__':
    main()
