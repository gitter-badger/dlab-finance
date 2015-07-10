# This file currently depends on python 3.3+

from zipfile import ZipFile
from datetime import datetime

from pytz import timezone
import numpy as np
from numpy.lib import recfunctions
import tables as tb

# Note that the '|' character means byte order doesn't matter, which it never
# will for "bytes" (which is what 'S' stands for - it doesn't stand for
# "string")
initial_dtype = [('Time', 'S9'),  # HHMMSSmmm, should be in Eastern Time (ET)
                 # ('hour', '|S2'),
                 # ('minute', '|S2'),
                 # ('second', '|S2'),
                 # ('msec', '|S3'),
                 ('Exchange', 'S1'),
                 # Wikipedia has a nice explanation of symbols here:
                 # https://en.wikipedia.org/wiki/Ticker_symbol
                 ('Symbol_root', 'S6'),
                 ('Symbol_suffix', 'S10'),
                 ('Bid_Price', 'S11'),  # 7.4 (fixed point)
                 ('Bid_Size', 'S7'),
                 ('Ask_Price', 'S11'),  # 7.4
                 ('Ask_Size', 'S7'),
                 ('Quote_Condition', 'S1'),
                 # Market_Maker ends up getting discarded, it should always be b'    '
                 ('Market_Maker', 'S4'),
                 ('Bid_Exchange', 'S1'),
                 ('Ask_Exchange', 'S1'),
                 ('Sequence_Number', 'S16'),
                 ('National_BBO_Ind', 'S1'),
                 ('NASDAQ_BBO_Ind', 'S1'),
                 ('Quote_Cancel_Correction', 'S1'),
                 ('Source_of_Quote', 'S1'),
                 ('Retail_Interest_Indicator_RPI', 'S1'),
                 ('Short_Sale_Restriction_Indicator', 'S1'),
                 ('LULD_BBO_Indicator_CQS', 'S1'),
                 ('LULD_BBO_Indicator_UTP', 'S1'),
                 ('FINRA_ADF_MPID_Indicator', 'S1'),
                 ('SIP_generated_Message_Identifier', 'S1'),
                 ('National_BBO_LULD_Indicator', 'S1'),
                 ('newline', 'S2')]

# This could be computed from the above bytes? - Would be easy if we change the
# above to a more convenient format.
BYTES_PER_LINE = 98

# Justin and Pandas (I think) use time64, as does PyTables.
# We could use msec from beginning of day for now in an int16
# (maybe compare performance to datetime64? But dates should compress very well...)
time_col = 'Time'

convert_dtype = [
               ('Bid_Price', np.float64),
               ('Bid_Size', np.int32),
               ('Ask_Price', np.float64),
               ('Ask_Size', np.int32),
               # ('Market_Maker', np.int8),  # This is not currently used, and should always be b'    '
               ('Sequence_Number', np.int64),
               # ('National_BBO_Ind', np.int8),  # The _Ind fields are actually categorical - leaving as strings
               # ('NASDAQ_BBO_Ind', np.int8),
              ]

passthrough_strings = ['Exchange',
                     'Symbol_root',
                     'Symbol_suffix',
                     'Quote_Condition',
                     'Bid_Exchange',
                     'Ask_Exchange',
                     'National_BBO_Ind',  # The _Ind fields are actually categorical - leaving as strings
                     'NASDAQ_BBO_Ind',
                     'Quote_Cancel_Correction',
                     'Source_of_Quote',
                     'Retail_Interest_Indicator_RPI',
                     'Short_Sale_Restriction_Indicator',
                     'LULD_BBO_Indicator_CQS',
                     'LULD_BBO_Indicator_UTP',
                     'FINRA_ADF_MPID_Indicator',
                     'SIP_generated_Message_Identifier',
                     'National_BBO_LULD_Indicator']

# Lifted from blaze.pytables
def dtype_to_pytables(dtype):
    """ Convert NumPy dtype to PyTable descriptor
    Examples
    --------
    >>> from tables import Int32Col, StringCol, Time64Col
    >>> dt = np.dtype([('name', 'S7'), ('amount', 'i4'), ('time', 'M8[us]')])
    >>> dtype_to_pytables(dt)  # doctest: +SKIP
    {'amount': Int32Col(shape=(), dflt=0, pos=1),
     'name': StringCol(itemsize=7, shape=(), dflt='', pos=0),
     'time': Time64Col(shape=(), dflt=0.0, pos=2)}
    """
    d = {}
    for pos, name in enumerate(dtype.names):
        dt, _ = dtype.fields[name]
        if issubclass(dt.type, np.datetime64):
            tdtype = tb.Description({name: tb.Time64Col(pos=pos)}),
        else:
            tdtype = tb.descr_from_dtype(np.dtype([(name, dt)]))
        el = tdtype[0]  # removed dependency on toolz -DJC
        getattr(el, name)._v_pos = pos
        d.update(el._v_colobjects)
    return d

def record_len_to_last_column(initial_dtype):
    """
    initial_dtype of form:
    
    [('Time', 'S9'),
 ('Exchange', 'S1'),
 ....
 ('newline', 'S2')]
 
 Assumption is that the last field is a newline field that is present in all versions of BBO
    """
    
    cum_len = 0
    cum_lens = []
    flens = [(field, int(dtype[1:])) for (field, dtype) in initial_dtype]
    newline_len = flens[-1][1]

    for (i,(field, flen)) in enumerate(flens[:-1]):
        cum_len += flen
        cum_lens.append((cum_len+newline_len, i))

    return dict(cum_lens)
    
    

# The "easy" dtypes are the "not datetime" dtypes
easy_dtype = []
convert_dict = dict(convert_dtype)

for name, dtype in initial_dtype:
    if name in convert_dict:
        easy_dtype.append( (name, convert_dict[name]) )
    elif name in passthrough_strings:
        easy_dtype.append( (name, dtype) )

# PyTables will not accept np.datetime64, we hack below, but we use it to work
# with the blaze function above.
# We also shift Time to the end (while I'd rather maintain order), as it's more
# efficient for Dav given the technical debt he's already built up.
pytables_dtype = easy_dtype + [('Time', 'datetime64[ms]')]
pytables_desc = dtype_to_pytables( np.dtype(pytables_dtype) )

RECORD_LEN_TO_LAST_COLUMN_MAP = record_len_to_last_column(initial_dtype)  

# TODO HDF5 will be broken for now
class TAQ2Chunks:
    '''Read in raw TAQ BBO file, and return numpy chunks (cf. odo)'''

    def __init__(self, taq_fname, chunksize=1000000, process_chunk=False):
        self.taq_fname = taq_fname
        self.chunksize = chunksize
        self.process_chunk = process_chunk
        
        self.numlines = None
        self.year = None
        self.month = None
        self.day = None
        
        self.iter_ = self.convert_taq()
        
    def __len__(self):
        return self.numlines
    
    def __iter__(self):
         return self
        
    def __next__(self):
        return next(self.iter_)


    def convert_taq(self):
        '''Return a generator that yields chunks
        chunksize : int
            Number of rows in each chunk
        '''
        # The below doesn't work for pandas (and neither does `unzip` from the
        # command line). Probably want to use something like `7z x -so
        # my_file.zip 2> /dev/null` if we use pandas.
        with ZipFile(self.taq_fname) as zfile:
            for inside_f in zfile.filelist:
                # The original filename is available as inside_f.filename
                self.infile_name = inside_f.filename

                with zfile.open(inside_f.filename) as infile:
                    first = infile.readline()
                    bytes_per_line = len(first)
                    
                    dtype = (initial_dtype[:RECORD_LEN_TO_LAST_COLUMN_MAP[bytes_per_line]+1] + [initial_dtype[-1]])

                    # You need to use bytes to split bytes
                    # some files (probably older files do not have a record count)
                    try:
                        dateish, numlines = first.split(b":")
                        self.numlines = int(numlines)
                        # Get dates to combine with times later
                        # This is a little over-trusting of the spec...
                        self.month = int(dateish[2:4])
                        self.day = int(dateish[4:6])
                        self.year = int(dateish[6:10])
                    
                    except:
                        pass

                    
                    if self.process_chunk:
                        yield from self.chunks(self.numlines, infile, self.chunksize)  # noqa
                    else:
                        more_bytes = True
                        while (more_bytes):
                            raw_bytes = infile.read(bytes_per_line * self.chunksize)
                            all_strings = np.ndarray(len(raw_bytes) // bytes_per_line, 
                                                     buffer=raw_bytes, dtype=dtype)
                            
                            if raw_bytes:
                                yield (all_strings)
                            else:
                                more_bytes = False                            
                        

    def process_chunk(self, all_strings):
        # This is unnecessary copying
        easy_converted = all_strings.astype(easy_dtype)

        # These don't have the decimal point in the TAQ file
        for dollar_col in ['Bid_Price', 'Ask_Price']:
            easy_converted[dollar_col] /= 10000

        # Currently, there doesn't seem to be any utility to converting to
        # numpy.datetime64 PyTables wants float64's corresponding to the POSIX
        # Standard (relative to 1970-01-01, UTC)
        converted_time = [datetime(self.year, self.month, self.day,
                                   int(raw[:2]), int(raw[2:4]), int(raw[4:6]),
                                   # msec must be converted to  microsec
                                   int(raw[6:9]) * 1000,
                                   tzinfo=timezone('US/Eastern') ).timestamp()
                          for raw in all_strings['Time'] ]

        # More unnecessary copying
        records = recfunctions.append_fields(easy_converted, 'Time',
                                             converted_time, usemask=False)

        return records

    def chunks(self, numlines, infile, chunksize=None):
        '''Do the conversion of bytes to numpy "chunks"'''
        # Should do check on numlines to make sure we get the right number

        while(True):
            raw_bytes = infile.read(BYTES_PER_LINE * chunksize)
            if not raw_bytes:
                break
            # If we use asarray with this dtype, it crashes Python! (might not be true anymore)
            # ndarray gives 'S' arrays instead of chararrays (as recarray does)
            all_strings = np.ndarray(chunksize, buffer=raw_bytes, dtype=initial_dtype)

            # This approach doesn't work...
            # out[chunk_start:chunk_stop, 1:] = all_strings[:,1:-1]

            yield self.process_chunk(all_strings)

    # Everything from here down is HDF5 specific
    # def setup_hdf5(self, h5_fname_root, numlines):
    #     # We're using aggressive compression and checksums, since this will
    #     # likely stick around Stopping one level short of max compression -
    #     # don't be greedy.
    #     self.h5 = tb.open_file(h5_fname_root + '.h5', title=h5_fname_root,
    #                            mode='w', filters=tb.Filters(complevel=8,
    #                                                         complib='blosc:lz4hc',
    #                                                         fletcher32=True) )

    #     return self.h5.create_table('/', 'daily_quotes', description=pytables_desc, expectedrows=numlines)


    # def finalize_hdf5(self):
    #     self.h5.close()


    # def to_hdf5(self, numlines, infile, out, chunksize=None):
    #     '''Read raw bytes from TAQ, write to HDF5'''

    #     # Should I use a context manager here?
    #     h5_table = self.setup_hdf5(inside_f.filename, numlines)
    #     try:
    #         self.to_hdf5(numlines, infile, h5_table)
    #     finally:
    #         self.finalize_hdf5()

    #     # at some point, we might optimize chunksize. For now, assume PyTables is smart
    #     if chunksize is None:
    #         chunksize = out.chunkshape[0]

    #     for chunk in self.to_chunks(numlines, infile, chunksize):
    #         out.append(chunk)


if __name__ == '__main__':
    from sys import argv
    from glob import glob

    try:
        fname = argv[1]
    except IndexError:
        # Grab the first BBO file we can find
        fname = glob('../local_data/EQY_US_ALL_BBO_*.zip')[0]

    test_run = TAQ2Chunks(fname)
    test_run.to_hdf5()