import pandas as pd
import os
message_cols = []

DROPBOX_DIR = os.getenv('DROPBOX_DIR')
# NLP prep from excel:

def _desirable_filename(filename):
    return all([
        filename.endswith('.xlsx'),
        'Moderat' in filename,
        'JH' not in filename])

def get_mega_df():
    dfs = []
    for d, cts, files in os.walk(DROPBOX_DIR):
        if 'Completed Transcripts' not in d:
            continue
        if len(files) == 1 and _desirable_filename(files[0]):
            print files[0]
            xls = pd.ExcelFile(os.path.join(d, files[0]))
            dfs.append(xls.parse(xls.sheet_names[0]))
    return pd.concat(dfs)

def get_message_cols(df):
    cols = [x for x in df.columns if x.startswith('Message')]
    cols.sort(key=lambda v: int(v[v.find(' ') + 1:]))
    return cols


def assemble_messages(row, message_cols, value_filter=lambda v: True, prepend_sender=False):
    sequence = []
    for key in message_cols:
        cell = row[key]
        if isinstance(cell, unicode) and ':' in cell:
            if value_filter(cell):
                message = cell[cell.find(':') + 1:]
                if prepend_sender:
                    sequence += ['$visitor' if cell.startswith('Visitor') else '$agent']
                sequence += message.split()
        else:
            return sequence
