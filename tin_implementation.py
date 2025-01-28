# https://towardsdatascience.com/implementing-vision-transformer-vit-from-scratch-3e192c6155f0


from torch import nn

"""
Step 1 : Convert an image into multiple patches and vectorise it
"""
class PatchEmbeddings(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"] # output dimension of each patch after projection

        self.num_patches = (self.image_size//self.patch_size) ** 2
        
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride = self.patch_size )
        """
        The convolutional layer outputs a feature map where each “pixel” corresponds to a patch from the original image.
        Each patch is projected into a hidden_size-dimensional vector, effectively embedding the patch into a higher-dimensional space.
        """

    def forward(self,x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1,2)
        