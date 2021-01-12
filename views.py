from django.shortcuts import render
from django.views.generic import TemplateView
from .forms import HandwritingForm
from .handwriting import *
from .models import HandwritingModel
# Create your views here.

class Handwriting(TemplateView):
    template_name = 'sample/handwriting.html'

    def get(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)
        context['handwriting_form'] = HandwritingForm
        return self.render_to_response(context)

    def post(self, request, *args, **kwargs):
        context = self.get_context_data(**kwargs)

        fm = HandwritingForm(self.request.POST, self.request.FILES)
        image_name = self.request.FILES['image'].name

        if fm.is_valid():
            if not HandwritingModel.objects.filter(image__icontains=image_name).exists():
                fm.save()
            labels, imageurl = get_handwritten_words(image_name)
            labels = list(labels)
            context['predicted_image'] = imageurl
            context['labels'] = labels
            return self.render_to_response(context)
        else:
            return HttpResponse('Something went wrong')
