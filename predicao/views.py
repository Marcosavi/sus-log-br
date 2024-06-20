from django.shortcuts import render, redirect #type: ignore
from predicao.forms import PredicaoForm

def predicao_view(request):
    if request.method == 'POST':
        form = PredicaoForm(request.POST, request.FILES)
        if form.is_valid():
            predicao = form.save(commit=False)
            predicao.save()
            return redirect('inicial:index')
        else:
            # Print form errors to debug
            print(form.errors)
    else:
        form = PredicaoForm()
    return render(request, 'inicial:index', {'form': form})