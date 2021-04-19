import time

from django.views.decorators.csrf import csrf_exempt
from rest_framework import permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from model.api import get_plot_data, steps


@csrf_exempt
@api_view(http_method_names=['post'])
@permission_classes((permissions.AllowAny,))
def train(request):
    # reset steps
    steps[:] = []
    params = request.data
    res, rmse, r2 = get_plot_data(params)
    return Response({
        'rawData': res,
        'rmse': rmse,
        'r2': r2
    })


# noinspection PyUnusedLocal
@csrf_exempt
@api_view(http_method_names=['post'])
@permission_classes((permissions.AllowAny,))
def get_step(request):
    params = request.data
    while len(steps) < int(params['step']):
        time.sleep(1)
        continue
    return Response({'steps': steps})
