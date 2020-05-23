import requests


def to_raw(url):
    rawgh = 'raw.githubusercontent.com'
    gh_pos = 2
    blob_pos = 5
    sep = "/"

    temp_url = url.split('/')
    del temp_url[blob_pos]
    temp_url[gh_pos] = rawgh

    new_url = ""
    for i in range(0, len(temp_url) - 1):
        new_url += temp_url[i] + sep
    new_url += temp_url[len(temp_url) - 1]
    return new_url


def get_text_by_url(url):
    r = requests.get(to_raw(url))
    return r.text


if __name__ == '__main__':
    url = 'https://github.com/nnovielli/jabref/blob/master/src/main/java/org/jabref/gui/fieldeditors/TextInputControlBehavior.java'
    print(get_text_by_url(
        'https://github.com/nnovielli/jabref/blob/master/src/main/java/org/jabref/gui/fieldeditors/TextInputControlBehavior.java'))

