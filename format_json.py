import json

import paramparse


class Params:
    in_file = "/data/youtube_vis19/train.json"
    # in_file = "/data/youtube_vis19/valid.json"


# def hook(obj):
#     for key, value in obj.items():
#         pbar = tqdm(value)
#         if type(value) is list:
#             for _ in pbar:
#                 pbar.set_description("Loading " + str(key))
#     return obj


def main():
    params = Params()
    paramparse.process(params)

    in_file = params.in_file

    # Opening JSON file
    f = open(in_file)

    print('loading json...')
    data = json.load(f,
                     # object_hook=hook
                     )

    print()

    print('dumping json...')
    output_json = json.dumps(data, indent=4)

    out_file = in_file + '.out'
    with open(out_file, 'w') as f:
        print('saving to file...')
        f.write(output_json)


if __name__ == '__main__':
    main()
