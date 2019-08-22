
import json

def load_json(file_name):
    with open(file_name) as json_file:
        data = json.load(json_file)
        return data

def GET_COLUMNS(input_data=None):


    WIDE_CATE_COLS = []
    DEEP_EMBEDDING_COLS = []
    CONTINUOUS_COLS = []
    DEEP_SHARED_EMBEDDING_COLS = []

    ORIGIN_DEEP_SHARED_EMBEDDING_COLS = []
    for fea in input_data:
        if 'col_type' in fea:
            if isinstance(fea['col_type'],list):
                for col in fea['col_type']:
                    if col == 'WIDE_CATE_COLS':
                        WIDE_CATE_COLS.append((fea['name'], fea['bucket_size']))
                    if col == 'DEEP_EMBEDDING_COLS':
                        DEEP_EMBEDDING_COLS.append(
                            (fea['name'], fea['bucket_size'], fea['embedding_size'], fea['type']))
                    if col == 'CONTINUOUS_COLS':
                        CONTINUOUS_COLS.append(fea['name'])
                    if fea['col_type'] == 'DEEP_SHARED_EMBEDDING_COLS':
                        ORIGIN_DEEP_SHARED_EMBEDDING_COLS.append(
                            (fea['name'], fea['bucket_size'], fea['embedding_size'], fea['type'], fea['shared_flag']))
            else:
                if fea['col_type'] == 'WIDE_CATE_COLS':
                    WIDE_CATE_COLS.append((fea['name'], fea['bucket_size']))
                if fea['col_type'] == 'DEEP_EMBEDDING_COLS':
                    DEEP_EMBEDDING_COLS.append((fea['name'], fea['bucket_size'], fea['embedding_size'], fea['type']))
                if fea['col_type'] == 'CONTINUOUS_COLS':
                    CONTINUOUS_COLS.append(fea['name'])
                if fea['col_type'] == 'DEEP_SHARED_EMBEDDING_COLS':
                    ORIGIN_DEEP_SHARED_EMBEDDING_COLS.append(
                        (fea['name'], fea['bucket_size'], fea['embedding_size'], fea['type'], fea['shared_flag']))

    print("ORIGIN_DEEP_SHARED_EMBEDDING_COLS:", ORIGIN_DEEP_SHARED_EMBEDDING_COLS)
    shared_flags = set()
    for _, _, _, _, flag in ORIGIN_DEEP_SHARED_EMBEDDING_COLS:
        shared_flags.add(flag)

    for c_flag in shared_flags:
        names = []
        bucket_sizes = []
        embedding_sizes = []
        types = []
        for name, bucket_size, embedding_size, type, flag in ORIGIN_DEEP_SHARED_EMBEDDING_COLS:
            if c_flag == flag:
                names.append(name)
                bucket_sizes.append(bucket_size)
                embedding_sizes.append(embedding_size)
                types.append(type)
        DEEP_SHARED_EMBEDDING_COLS.append((names, bucket_sizes[0], embedding_sizes[0], types[0], c_flag))

    print("DEEP_SHARED_EMBEDDING_COLS:", DEEP_SHARED_EMBEDDING_COLS)
    print("WIDE_CATE_COLS:", WIDE_CATE_COLS)
    print('CONTINUOUS_COLS:', CONTINUOUS_COLS)
    print('DEEP_EMBEDDING_COLS:', DEEP_EMBEDDING_COLS)

    WIDE_CROSS_COLS = (('pv', 'created_time', 140),
                       ('pv', 'g_created_time', 140),
                       ('class2', 'class2', 250))
    return WIDE_CATE_COLS,DEEP_EMBEDDING_COLS,CONTINUOUS_COLS,DEEP_SHARED_EMBEDDING_COLS,WIDE_CROSS_COLS
