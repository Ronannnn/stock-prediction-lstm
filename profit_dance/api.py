from django.views.decorators.csrf import csrf_exempt
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from model.api import get_plot_data


@csrf_exempt
@api_view(http_method_names=['post'])
@permission_classes((permissions.AllowAny,))
def train(request):
    params = request.data
    return Response({'rawData': get_plot_data(params['stockCode'])})
