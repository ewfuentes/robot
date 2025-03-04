from tensorboard import program


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str)
    args = parser.parse_args()
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', args.logdir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    while True:
        pass