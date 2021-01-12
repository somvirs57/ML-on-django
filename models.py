from django.db import models
import os

# Create your models here.
def handwriting_img_path(instance, filename):
    upload_to = 'user_uploaded'  #.format(str(instance.image).split('.')[0])
    ext = filename.split('.')[-1]
    # get filename
    if instance.image:
        filename = '{}.{}'.format(str(instance.image).split('.')[0], ext)
    return os.path.join(upload_to, filename)

class HandwritingModel(models.Model):
    image = models.ImageField(upload_to=handwriting_img_path, null=True, blank=True)

    def __str__(self):
        return str(self.image).split('/')[-1]
