#!/usr/bin/env python

# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
#
# See: https://docs.python.org/3/library/cmd.html
######################################################################
import argparse
import cmd
import logging
import sys

logging.basicConfig()
logger = logging.getLogger(__name__)

# Uncomment me to see which REPL commands are being run!
# logger.setLevel(logging.DEBUG)

from util import load_together_client, stream_llm_to_console, DEFAULT_STOP
from agent import react_agent, enhanced_agent
from agent import user_database, showtime_database, ticket_database, request_database  

import pprint

# Modular ASCII font from http://patorjk.com/software/taag/
HEADER = """Welcome to Stanford CS124's
 _______  _______       ___________
|       ||   _   |      \          \\
|    _  ||  |_|  | ____  ------    /
|   |_| ||       ||____|      /   /
|    ___||       |           /   /       
|   |    |   _   |          /   /      
|___|    |__| |__|         /___/

 _______  __   __  _______  _______  _______  _______  _______  __
|       ||  | |  ||   _   ||       ||  _    ||       ||       ||  |
|       ||  |_|  ||  |_|  ||_     _|| |_|   ||   _   ||_     _||  |
|       ||       ||       |  |   |  |       ||  | |  |  |   |  |  |
|      _||       ||       |  |   |  |  _   | |  |_|  |  |   |  |__|
|     |_ |   _   ||   _   |  |   |  | |_|   ||       |  |   |   __
|_______||__| |__||__| |__|  |___|  |_______||_______|  |___|  |__|
"""

description = 'Simple Read-Eval-Print-Loop that handles the input/output ' \
              'part of the conversational agent '


class REPL(cmd.Cmd):
    """Simple REPL to handle the conversation with the chatbot."""
    prompt = '> '
    doc_header = ''
    misc_header = ''
    undoc_header = ''
    ruler = '-'

    def __init__(self):
        super().__init__()

        self.agent = react_agent # TODO: replace with enhanced_agent to test the enhanced agent
        self.name = "Movie Ticket Agent"
        self.bot_prompt = '\001\033[96m\002%s> \001\033[0m\002' % self.name
        self.intro = '\n' + self.bot_prompt + \
                     "Hello! I'm the Movie Ticket Agent. How can I help you today?"

        # Mapping for quick printing
        self.agent_dbs = {
            'user_database': user_database,
            'ticket_database': ticket_database,
            'request_database': request_database
        }
        

    def cmdloop(self, intro=None):
        logger.debug('cmdloop(%s)', intro)
        return super().cmdloop(intro)

    def preloop(self):
        logger.debug('preloop(); Movie Ticket Agent created and loaded')
        print(HEADER)

    def postloop(self):
        goodbye = "Thank you for using the Movie Ticket Agent. Have a great day!"
        print(goodbye)

    def onecmd(self, s):
        logger.debug('onecmd(%s)', s)
        if s:
            return super().onecmd(s)
        else:
            return False  # Continue processing special commands.

    def emptyline(self):
        logger.debug('emptyline()')
        return super().emptyline()

    def default(self, line):
        logger.debug('default(%s)', line)
        # Support for 'print {db}'
        # Recognize "print user_database" etc.
        line_stripped = line.strip()
        printables = ["user_database", "showtime_database", "ticket_database", "request_database"]
        if line_stripped == ":quit":
            return True
        elif any(line_stripped == f"print {db}" for db in printables):
            # identify db name
            _, dbname = line_stripped.split()
            dbobj = self.agent_dbs.get(dbname)
            if dbobj is not None:
                print(f"\nPrinting {dbname}:")
                pprint.pprint(dbobj)
            else:
                print(f"Unknown database: {dbname}")
        else:
            response = self.agent(user_request=line) 
            print(response)


if __name__ == '__main__':
    #########################
    # ADDED FOR TESTING     #
    #########################
    class Tee(object):
        # Modified from
        # https://stackoverflow.com/questions/34366763/input-redirection-with-python
        def __init__(self, input_handle, output_handle):
            self.input = input_handle
            self.output = output_handle

        def readline(self):
            result = self.input.readline()
            self.output.write(result)
            self.output.write('\n')
            self.output.flush()

            return result

        # Forward all other attribute references to the input object.
        def __getattr__(self, attr):
            return getattr(self.input, attr)


    if not sys.stdin.isatty():
        sys.stdin = Tee(input_handle=sys.stdin, output_handle=sys.stdout)

    #########################
    # END TESTING CODE      #
    #########################
    repl = REPL()
    repl.cmdloop()
