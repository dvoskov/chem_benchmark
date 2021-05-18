import os, sys

# keep original stream to be able to abort redirection
original_stdout = os.dup(1)
# redirected stream will be stored here
log_stream = [1]

# low-level redirection works for all messages from both Python and C++, printf or std::cout
def redirect_all_output(log_file, append = True):
    if log_file == '':
        log_file = os.devnull
    if append:
        log_stream[0] = open(log_file, "a+")
    else:
        log_stream[0] = open(log_file, "w")

    os.dup2(log_stream[0].fileno(), sys.stdout.fileno())

def abort_redirection():
    os.dup2(original_stdout, sys.stdout.fileno())
    log_stream[0].close()


