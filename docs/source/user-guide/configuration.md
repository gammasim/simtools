(configuration)=

# Configuration

Applications in simtools can be configured by the following four approaches, which are all equivalent:

1. command-line arguments;
2. configuration files (in yaml format);
3. configuration dictionary when calling the {ref}`Configurator <configurationconfigurator>` class;
4. environment variables.

To illustrate this, the example below sets the path pointing towards the directory for all data products.

Set the output directory using a command-line argument:

```console
python applications/<application_name> --output_path <path name>
```

Set the output directory using a configuration file in yaml format:

```yaml
config_file: <path name>
```

Load the yaml configuration file into the application with:

```console
python applications/<application_name> --config <my_config.yml>
```

Configuration parameter read from a environmental variable:

```console
EXPORT SIMTOOLS_OUTPUT_PATH="<path name>"
```

Configuration methods can be combined; conflicting configuration settings raise an Exception.
Configuration parameters are generally expected in lower-case snake-make case.
Configuration parameters for each application are printed to screen when executing the application with the `--help` option.
Parameters with the same functionality are named consistently the same among all applications.
