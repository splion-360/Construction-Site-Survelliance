import telepot
import emoji
import time
#Constants
token = '2035662141:AAFxj_dpZZhGsyYF-5aUwYCNMmKFrjAp2_k'
receiver_id = 2020703840
bot = telepot.Bot(token)

def SendInfo():
    msg = 'Boss !! I have detected a face '
    bot.sendMessage(receiver_id, msg)
    #image_path = photo
    #bot.sendPhoto(receiver_id, photo=open(image_path, 'rb'))
    time.sleep(6)
    dict = bot.getUpdates(offset=10000000000001)
    text = dict[-1]['message']['text']

    while (text != 'Stupid' and text != 'Cool'):
        dict = bot.getUpdates()
        text = dict[-1]['message']['text']
        continue

    if text == 'Stupid':
        bot.sendMessage(receiver_id, emoji.emojize(':smiling_face_with_tear:'))
        bot.sendMessage(receiver_id, 'Will do better, boss')

    elif text == 'Cool':
        bot.sendMessage(receiver_id,emoji.emojize(':smiling_face_with_sunglasses:'))
        bot.sendMessage(receiver_id,'Thanks boss')

SendInfo()