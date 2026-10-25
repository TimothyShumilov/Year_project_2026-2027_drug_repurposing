"""One-page, minimally formatted practical-value report (bundled ReportLab)."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

ROOT=Path(__file__).resolve().parents[1]
font_pairs=[('/System/Library/Fonts/Supplemental/Arial.ttf','/System/Library/Fonts/Supplemental/Arial Bold.ttf'),
            ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
            ('C:/Windows/Fonts/arial.ttf','C:/Windows/Fonts/arialbd.ttf')]
font_regular,font_bold=next((a,b) for a,b in font_pairs if Path(a).exists() and Path(b).exists())
pdfmetrics.registerFont(TTFont('ReportArial',font_regular))
pdfmetrics.registerFont(TTFont('ReportArialBold',font_bold))
pdfmetrics.registerFontFamily('ReportArial',normal='ReportArial',bold='ReportArialBold')
body=ParagraphStyle('Body',fontName='ReportArial',fontSize=11,leading=15,
                    spaceAfter=10,textColor=colors.black,alignment=TA_LEFT)
title=ParagraphStyle('Title',parent=body,fontName='ReportArialBold',fontSize=16,leading=20,spaceAfter=15)

paragraphs=[
    ('Практическая ценность проекта',title),
    ('Проект прогнозирует силу взаимодействия малых молекул с белковыми мишенями. '
     'Пользователь - исследователь в области разработки лекарств, которому нужно выбрать '
     'перспективные взаимодействия для лабораторной проверки.',body),
    ('<b>Какую задачу решает.</b> Экспериментальная проверка большого числа молекул на разных '
     'белках требует времени и ресурсов. Модель помогает сузить список: для известного препарата '
     'предлагает дополнительные мишени, а для выбранной мишени ранжирует доступные соединения. '
     'Это может поддержать поиск новых применений существующих препаратов.',body),
    ('<b>Как будет использоваться.</b> Пользователь передаёт структуру молекулы в формате SMILES '
     'и получает прогноз аффинности для поддерживаемого набора мишеней. Результат сопровождается '
     'описанием версии данных, области применимости и ограничений. Прогноз связывания служит '
     'гипотезой для эксперимента: он не устанавливает лечебный эффект или безопасную дозировку.',body),
    ('<b>Как оценивать качество.</b> На заранее отложенных молекулах сравниваем ошибку прогноза '
     'с простым Random Forest. Основная проверка использует химические каркасы, отсутствующие '
     'в обучении. Заказчик также может провести слепую лабораторную проверку отобранных '
     'взаимодействий и оценить долю подтверждённых кандидатов и сокращение числа экспериментов.',body),
    ('<b>Ограничения.</b> Публикуемые измерения неравномерно покрывают белки и соединения. '
     'Данные разных лабораторий могут противоречить друг другу; отсутствие измерения не означает '
     'отсутствия взаимодействия. На необычных структурах качество может снижаться. '
     'Текущая постановка поддерживает только изученные мишени. Перенос на новые белки '
     'потребует признаков белка и отдельного тестирования.',body),
    ('<b>Реалистичность.</b> Прототип сервиса реализуем на открытых данных ChEMBL и обычных '
     'инструментах машинного обучения. Потребуются проверка входных структур, фиксация версий '
     'данных и моделей, соблюдение лицензий и защита загружаемых пользователями молекул. '
     'Клиническое применение выходит за пределы проекта.',body),
    ('<b>Исследовательская ценность.</b> Проверяем, помогает ли совместное обучение по нескольким '
     'мишеням переносить знания на задачи с небольшим количеством измерений. Зафиксированные '
     'данные, разбиения и бейзлайн позволяют воспроизводимо сравнивать последующие модели.',body),
]
output=ROOT/'output/pdf/practical_value.pdf';output.parent.mkdir(parents=True,exist_ok=True)
document=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=52,leftMargin=52,
                           topMargin=45,bottomMargin=45,title='Практическая ценность проекта прогнозирования аффинности',
                           author='Годовой проект')
document.build([Paragraph(text,style) for text,style in paragraphs])
print(output)
