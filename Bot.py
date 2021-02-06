import telebot
import os
from PIL import Image


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

import copy

# Style Transfer part
imsize = 224
loader = transforms.Compose([
    transforms.Resize(imsize),  # нормируем размер изображения
    transforms.CenterCrop(imsize),
    transforms.ToTensor()])  # превращаем в удобный формат

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()  # это константа. Убираем ее из дерева вычеслений
        self.loss = F.mse_loss(self.target, self.target)  # to initialize with something

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    batch_size, h, w, f_map_num = input.size()  # batch size(=1)
    # b=number of feature maps
    # (h,w)=dimensions of a feature map (N=h*w)
    # print(h,w, f_map_num)

    features = input.view(batch_size * h, w * f_map_num)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(batch_size * h * w * f_map_num)


class StyleLoss(nn.Module):
    def __init__(self, target_feature, mask):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)  # to initialize with something
        # print(target_feature.size())
        n = target_feature.size()[1]
        m = target_feature.size()[2]
        loader = transforms.Compose([
            transforms.Resize(m),  # нормируем размер изображения
            transforms.CenterCrop(m)])  # превращаем в удобный формат
        mask1 = loader(mask)
        # print(mask1.size())
        self.mask = torch.cat([torch.narrow(mask1, 1, 0, 1)] * n, dim=1)
        # print(self.mask.size())

    def forward(self, input):
        # print(input.size())
        G = gram_matrix(input * self.mask)
        self.loss = F.mse_loss(G, self.target)
        return input

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

cnn = models.vgg19(pretrained=True).features.to(device).eval()


def get_style_model_and_losses(cnn, normalization_mean, normalization_std, content_img,
                               style_img1, style_img2=None, mask=None,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses1 = []
    style_losses2 = []
    if mask == None:
        w = content_img.size()[-1]
        mask = torch.ones([1, 3, w, w])
        inverted_mask = torch.ones([1, 3, w, w])
    else:
        mask = mask
        #inverted_mask = inverted_mask_img

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            # Переопределим relu уровень
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature1 = model(style_img1).detach()
            style_loss1 = StyleLoss(target_feature1, mask)
            style_loss2 = 0
            model.add_module("style_loss1_{}".format(i), style_loss1)
            style_losses1.append(style_loss1)
            if style_img2 != None:
                target_feature2 = model(style_img2).detach()
                style_loss2 = StyleLoss(target_feature2, inverted_mask)
                model.add_module("style_loss2_{}".format(i), style_loss2)
            style_losses2.append(style_loss2)
    # now we trim off the layers after the last content and style losses
    # выбрасываем все уровни после последенего styel loss или content loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses1, style_losses2, content_losses

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    # добоваляет содержимое тензора катринки в список изменяемых оптимизатором параметров
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                           content_img, input_img, style_img1, style_img2=None, mask=None, num_steps=500,
                           style_weight=100000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses1, style_losses2, content_losses = get_style_model_and_losses(cnn,
                                                                                         normalization_mean,
                                                                                         normalization_std, content_img,
                                                                                         style_img1, style_img2, mask)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
                # correct the values
                # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()

            model(input_img)

            style_score1 = 0
            style_score2 = 0
            content_score = 0

            for sl in style_losses1:
                style_score1 += sl.loss
            if 0 not in style_losses2:
                for sl in style_losses2:
                    style_score2 += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            # взвешивание ощибки
            style_score1 *= style_weight
            style_score2 *= style_weight
            content_score *= content_weight

            loss = style_score1 + style_score2 + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                if 0 not in style_losses2:
                    print('Style 1 Loss : {:4f} Style 2 Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score1.item(), style_score2.item(), content_score.item()))
                else:
                    print('Style Loss : {:4f}  Content Loss: {:4f}'.format(
                            style_score1.item(), content_score.item()))
                print()

            return style_score1 + style_score2 + content_score

        optimizer.step(closure)

        # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img

# Bot part
bot = telebot.TeleBot("1431041965:AAEICsH9QB3t39bEEpYnI5-s3wBP_0tCnYI")

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Hi! Send me two pictures and I will do style transfer! First - content image, second - style image. Don't mix up!")

@bot.message_handler(content_types=['text'])
def unexpected_message(message):
    bot.reply_to(message, "I don't understand you. I am just a Style transfer bot - send me your content Image.")

@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    try:
        file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        images = [f for f in os.listdir('/home/anastasia/Downloads/tgbot/photos/'+str(message.from_user.id))]
        src='/home/anastasia/Downloads/tgbot/'+file_info.file_path
        with open(src, 'wb') as new_file:
            new_file.write(downloaded_file)
        os.rename(src, '/home/anastasia/Downloads/tgbot/photos/file_{}.jpg'.format(len(images)))
        if not os.path.exists('/home/anastasia/Downloads/tgbot/photos/'+ str(message.from_user.id)):
            os.makedirs('/home/anastasia/Downloads/tgbot/photos/'+str(message.from_user.id))
        os.replace('/home/anastasia/Downloads/tgbot/photos/file_{}.jpg'.format(len(images)),
                   '/home/anastasia/Downloads/tgbot/photos/{}/file_{}.jpg'.format(message.from_user.id, len(images)))
        bot.reply_to(message, "Received image!")
        images = [f for f in os.listdir('/home/anastasia/Downloads/tgbot/photos/'+str(message.from_user.id))]
        if len(images) == 1:
            bot.send_message(message.from_user.id, 'Got 1! Now send me your style image!')
        if len(images) == 2:
            bot.send_message(message.from_user.id,'Got 2! Processing your images! (It might take time)')
            content_img = image_loader("/home/anastasia/Downloads/tgbot/photos/{}/file_0.jpg".format(message.from_user.id))
            style_img = image_loader("/home/anastasia/Downloads/tgbot/photos/{}/file_1.jpg".format(message.from_user.id))
            input_img = content_img.clone()
            output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                         content_img, input_img, style_img, num_steps=500)
            save_image(output, '/home/anastasia/Downloads/tgbot/photos/{}/output.jpg'.format(message.from_user.id))
            images+=['output.jpg']
            bot.send_photo(message.from_user.id, photo=open('/home/anastasia/Downloads/tgbot/photos/{}/output.jpg'.format(message.from_user.id), 'rb'))
            for f in images:
                os.remove(os.path.join('/home/anastasia/Downloads/tgbot/{}/photos/{}'.format(message.from_user.id, f)))
            os.rmdir('/home/anastasia/Downloads/tgbot/{}/photos/{}'.format(message.from_user.id))

    except Exception as e:
        bot.reply_to(message,e )

bot.polling()
