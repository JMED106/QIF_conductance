import argparse

try:
    import yaml
except:
    raise ImportError(
        'Istall pyyaml package. In debian based systems:\n'
        '\t # apt-get install python-yaml\n or\n\t pip install PyYAML ')
import os
import sys
import datetime
import logging.config
try:
    from colorlog import ColoredFormatter
except:
    raise ImportError('Install colorlog module. In debian based systems:\n'
                      '\t# apt-get install python-colorlog\nor\n\tpip install colorlog')

__author__ = 'Jose M. Esnaola Acebes'

""" General pourpose library for simulations.

    + Parsing.
    + Logging.
    + Data loading.
    + Data saving.

"""



class Options:
    def __init__(self):
        pass


def parser_init():
    """ Function to handle arguments from CLI:
        First Parsing -  We parse optional configuration files.
    """

    pars = argparse.ArgumentParser(add_help=False)
    pars.add_argument('-f', '--file', default="conf.txt", dest='-f', metavar='<file>')
    pars.add_argument('-db', '--debug', default="DEBUG", dest='db', metavar='<debug>',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    # Check for arguments matching the latter options
    args = pars.parse_known_args()
    conf_file = vars(args[0])['-f']  # Configuration file (if any)
    debug_level = vars(args[0])['db']
    hlp = False
    for k, op in enumerate(args[1]):
        if op == '-h' or op == '--help':
            hlp = True
    return conf_file, debug_level, args, hlp


def parser(config_file, arguments=None, description=None, usage=None):
    """ Function to handle arguments from CLI:
        Second Parsing -  We parse simulation parameters.
        :param config_file: YAML format. See config_doc variable and YAML documentation for more help.
        :param arguments: if there is a previous parsing, introduce the arguments here.
        :param description: Description of the script/program (string)
        :param usage:  How to use the script. (string)
    """
    log.debug('Starting second parsing.')
    config_doc = """
        Block example title:
          -a --argument:
            description: "Example of how to introduce list arguments."
            default:     [1]
            name:        "<argument>"
            choices:     ~
          -anthr --another:
            description: "Another example, for float arguments."
            default:     0.0
            name:        "<another>"
            choices:     ~
        Another Block example title:
          -s --string:
            description: "String argument example with choices."
            default:     'string'
            name:        "<string>"
            choices:     ['string1', 'string2']
          -b --boolean:
            description: "Boolean argument example with choices."
            default:     False
            name:        "<bool>"
            choices:     [False, True]
    """

    # Opening the configuration file to load parameters
    options = None
    ops = Options()
    try:
        options = yaml.load(file(config_file, 'rstored'))
    except IOError:
        log.error("The configuration file '%s' is missing" % config_file)
        log.info("No configuration loaded.")
        options = yaml.load(config_doc)
    except yaml.YAMLError, exc:
        log.error("Error in configuration file:", exc)
        exit(-1)

    # Loading parameters from the 'options' dictionary and CLI options
    if usage is None:
        usage = 'python %s [-O <options>]' % sys.argv[0]
    pars = argparse.ArgumentParser(
        description=description,
        usage=usage)
    for group in options:
        gr = pars.add_argument_group(group)
        for key in options[group]:
            flags = key.split()
            args = options[group]
            if isinstance(args[key]['default'], bool):
                gr.add_argument(*flags, default=args[key]['default'], help=args[key]['description'], dest=flags[0][1:],
                                action='store_true')
            elif isinstance(args[key]['default'], list):
                tipado = type(args[key]['default'][0])
                gr.add_argument(*flags, default=args[key]['default'], help=args[key]['description'], dest=flags[0][1:],
                                metavar=args[key]['name'], type=tipado,
                                choices=args[key]['choices'], nargs='+')
            else:
                gr.add_argument(*flags, default=args[key]['default'], help=args[key]['description'], dest=flags[0][1:],
                                metavar=args[key]['name'], type=type(args[key]['default']),
                                choices=args[key]['choices'])

    # We parse command line arguments:
    if arguments:
        arg = arguments[1]
        opts = pars.parse_args(arg)
        args = pars.parse_args(arg, namespace=ops)
    else:
        opts = pars.parse_args()
        args = pars.parse_args(namespace=ops)
    return opts, args


def log_conf(db, config_file=None, name='simulation', logdir='./log'):
    """ Logging configuration
    :param db: debugging level, must be an attribute in logging. See help(logging).
    :param config_file: external logging configuration file for handlers configuration.
    :param name: name of the logger.
    :param logdir: Directory where the log file is stored.
    """
    global log
    logging_doc = """
        version: 1
        formatters:
          simple:
            format: "[%(levelname)-8.8s]:%(name)-20.20s:%(funcName)-10.10s:\n\t%(message)s"
        handlers:
          console:
            class: logging.StreamHandler
            level: DEBUG
            formatter: simple
            stream: ext://sys.stdout
          file:
            class: logging.FileHandler
            level: DEBUG
            formatter: simple
            filename: 'log/simulation.log'
        loggers:
          simulation:
            level: DEBUG
            handlers: [console, file]
            propagate: no
        root:
          level: DEBUG
          handlers: [console, file]
    """

    # Setting debug level
    debug = getattr(logging, db.upper(), None)
    # Check the value
    if not isinstance(debug, int):
        raise ValueError('Invalid log level: %s' % db)
    # Check logging folder (default is log)
    cwd = os.getcwd()
    if not os.path.exists(logdir):
        try:
            os.mkdir(logdir)
        except:
            raise IOError('Path %s/%s does not exist.' % (cwd, logdir))
    filename = "%s/%s.log" % (logdir, name)
    if os.path.exists(filename):
        f = file(filename, 'a+')
    else:
        f = file(filename, 'w+')
    day, hour = now()
    f.write("\n[%s\t%s]\n" % (day, hour))
    f.write("-------------------------\n")
    f.close()

    # Output format
    logformat = "%(log_color)s[%(levelname)-7.8s]%(reset)s %(name)-12.12s:%(funcName)-8.8s: " \
                "%(log_color)s%(message)s%(reset)s"
    formatter = ColoredFormatter(logformat, log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    })

    # General Configuration
    if config_file:
        logging.config.dictConfig(yaml.load(file(config_file, 'rstored')))
    else:
        logging.config.dictConfig(yaml.load(logging_doc))

    # Handler
    handler = logging.root.handlers[0]
    handler.setLevel(debug)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    log = logging.getLogger('sconf')
    log.debug("Logger succesfully set up. Starting debuging, level: %s" % db)

    return logger


def now(daysep=', ', hoursep=':'):
    """ Returns datetime """
    _now = datetime.datetime.now().timetuple()[0:6]
    sday = daysep.join(map(str, _now[0:3]))
    shour = hoursep.join(map(str, _now[3:]))
    return sday, shour
