import click

@click.group()
def main():
    pass


@main.command("preprocess_classification")
@click.option("--output-dir", type=str, required=True, help="Path to the output directory", default="dataset/sard_pose")
def preprocess_classification(output_dir):
    from sarfusion.data.preprocess import generate_pose_classification_dataset
    generate_pose_classification_dataset(output_dir)
 

@main.command("preprocess_wisard")
def preprocess_wisard():
    from sarfusion.data.preprocess import wisard_to_yolo_dataset
    wisard_to_yolo_dataset("dataset/WiSARD")    
    
    
@main.command("experiment")
@click.option(
    "--parameters", default="parameters.yaml", help="Path to the parameters file"
)
@click.option(
    "--parallel", default=False, help="Run the experiments in parallel", is_flag=True
)
@click.option(
    "--only-create",
    default=False,
    help="Creates params files with running them",
    is_flag=True,
)
def experiment(parameters, parallel, only_create):
    from sarfusion.experiment.experiment import experiment as run_experiment
    run_experiment(param_path=parameters, parallel=parallel, only_create=only_create)


@main.command("run")
@click.option(
    "--parameters", default="parameters.yaml", help="Path to the parameters file"
)
def run(parameters):
    from sarfusion.experiment.experiment import run as run_single
    run_single(param_path=parameters)
    
    
if __name__ == "__main__":
    main()