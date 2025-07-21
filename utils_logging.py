import logging

def setup_logging(logfile='summarizer_app.log'):
    logging.basicConfig(filename=logfile, level=logging.ERROR, 
                        format='%(asctime)s %(levelname)s %(message)s') 