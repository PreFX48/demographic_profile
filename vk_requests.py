import requests
import json


def scanCommunity (communityId, number):
    VK_API = 'http://api.vk.com/method/'
    posts = []
    count = 0
    while count < number:
        postsResponse = requests.get(VK_API+'wall.get',
                {'owner_id': '-' + str(communityId), 'offset': count, 'count': min(100, number-count)}).json()['response']
        if len(postsResponse) == 1:
            break
        posts += list(map(lambda post: post['id'], postsResponse[1:]))
        count += len(postsResponse) - 1

    comments = []
    for post in posts:
        count = 0
        while True:
            commentsResponse = requests.get(VK_API+'wall.getComments',
                {'owner_id': '-' + str(communityId), 'post_id': post,
                 'offset': count, 'count': 100}).json()['response']
            if len(commentsResponse) == 1:
                break
            comments += list(map(lambda comment: (comment['from_id'], comment['text']), commentsResponse[1:]))
            count += len(commentsResponse) - 1

    authors = {}
    commentNumber = 1
    for comment in comments:
        if commentNumber % 25 == 0:
            print('Comment #' + str(commentNumber), 'out of', len(comments))
        if comment[0] not in authors:
            authorInfo = requests.get(VK_API+'users.get',
                   {'user_ids':comment[0], 'fields': 'bdate'}).json()['response']
            if len(authorInfo) != 0:
                if 'bdate' in authorInfo[0]:
                    if authorInfo[0]['bdate'].find('.', authorInfo[0]['bdate'].find('.') + 1) != -1:
                        authorInfo[0]['user_comments'] = [comment[1]]
                        authors[comment[0]] = authorInfo[0]
        else:
            authors[comment[0]]['user_comments'].append(comment[1])
        commentNumber += 1


    with open('data.json', 'w') as fp:
        json.dump(list(authors.values()), fp)

# id сообщества Лентач - 29534144 (для дальнейшего использования)
# scanCommunity(29534144, 50)
