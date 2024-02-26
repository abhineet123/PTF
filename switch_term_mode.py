import paramparse
import json


@paramparse.process
class Params:
    json_path = ""
    dark_scheme = "Campbell"
    light_scheme = "One Half Light"


def main():
    # params = Params
    json_path = Params.json_path

    assert json_path, "json_path must be provided"

    json_dict = json.load(open(json_path, "r"))


    for profile in json_dict['profiles']['list']:

        try:
            color_scheme = profile['colorScheme']
        except KeyError:
            continue

        if color_scheme == Params.dark_scheme:
            profile['colorScheme'] = Params.light_scheme
        elif color_scheme == Params.light_scheme:
            profile['colorScheme'] = Params.dark_scheme
        else:
            raise AssertionError(f'invalid color_scheme: {color_scheme}')

    with open(json_path, 'w') as f:
        output_json_data = json.dumps(json_dict, indent=4)
        f.write(output_json_data)


if __name__ == '__main__':
    main()
