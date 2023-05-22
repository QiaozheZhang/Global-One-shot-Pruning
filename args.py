import argparse
import sys
import yaml

global parser_args

class Args:
    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="my args")

        parser.add_argument(
            "--config",
            default='configs/FC12_MNIST.yml',
            help="Config file to use"
        )

        args = parser.parse_args()
        self.get_config(args)

        return args


    def get_config(self, parser_args):
        # load yaml file
        yaml_txt = open(parser_args.config).read()

        # override args
        loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
        print(f"=> Reading YAML config from {parser_args.config}")
        parser_args.__dict__.update(loaded_yaml)

    def get_args(self):
        global parser_args
        parser_args = self.parse_arguments()

args = Args()
args.get_args()