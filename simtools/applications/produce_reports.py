#!/usr/bin/python3

import logging
import numpy as np
import simtools.utils.general as gen
from pathlib import Path
from simtools.reporting.read_parameters import ParamDicts
from simtools.configuration import configurator
from simtools.io_operations import io_handler
from simtools.model.telescope_model import TelescopeModel


def _parse(label):
    """Parse command line configuration."""
    config = configurator.Configurator(
            label=label,
            description=(
                "Generate an html report for all the model parameter at a given version associated with a given telescope at a given site."
                ),
            )

    return config.initialize(db_config=True, simulation_model=["telescope"])



def generate_html_report(args_dict, data):
    '''
    A function to generate an html file to report the parameter values.
    '''

    io_handler_instance = io_handler.IOHandler()
    output_path = io_handler_instance.get_output_directory(Path(__file__).stem, sub_dir="html-reports")
    output_filename = f'{output_path}/{args_dict["telescope"]}_v_{args_dict["model_version"]}'

    # Start writing the HTML file
    with open(output_filename + '.html', 'w') as file:
        # HTML boilerplate
        file.write("<html><head><title>%s</title>\n" %output_filename)
        file.write("<style>\n")
        file.write("body { font-family: Arial, sans-serif; margin: 40px; }\n")
        file.write("h1 { color: #333; }\n")
        file.write("table { width: 100%; border-collapse: collapse; margin: 20px 0; }\n")
        file.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
        file.write("th { background-color: #f2f2f2; }\n")
        file.write("</style>\n")
        file.write("</head><body>\n")
        file.write("<h1>Parameter report for %s in model version %s<h1>" %(args_dict["telescope"], args_dict["model_version"]))

        # Write the section header to specify the telescope
        file.write("<h2>%s</h2>\n" %args_dict['telescope'])
        file.write("<hr>\n")

        # Start the table for displaying parameters
        file.write("<table>\n")
        file.write("<tr><th>Parameter</th><th>Value</th><th>Unit</th><th>Short Description</th></tr>\n")

        # Write each parameter to the table
        for item in data:
            #print('item: ', item)
            file.write(f"<tr><td>{item[0]}</td><td>{item[1]}</td><td>{item[2]}</td><td>{item[4]}</td></tr>\n")

        file.write("</table>\n")

    # End the HTML file
    with open(output_filename + '.html', 'a') as file:
        file.write("</body></html>\n")




def main(): 
    label = Path(__file__).stem
    print('label: ', label)
    args_dict, db_config = _parse(label)
    print(args_dict, db_config)

    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict["log_level"]))

    _io_handler = io_handler.IOHandler()
    output_dir = _io_handler.get_output_directory(label, sub_dir="html-reports")

    tel_model = TelescopeModel(
            site=args_dict["site"],
            telescope_name=args_dict["telescope"],
            model_version=args_dict["model_version"],
            label=label,
            mongo_db_config=db_config,
            )


    data = ParamDicts().get_telescope_param_data(tel_model)
    
    generate_html_report(args_dict, data) 

    logger.info(f"HTML report generated: {args_dict['telescope']} for model version {args_dict['model_version']}")


if __name__ == '__main__':
    main()
