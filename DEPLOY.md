# 🚀 Инструкция по деплою ForecastIQ на Streamlit Cloud

## Структура проекта
```
forecastiq/
├── app.py                  ← главный файл
├── requirements.txt        ← библиотеки
└── .streamlit/
    └── config.toml         ← тёмная тема
```

---

## Шаг 1 — Установить Git (если не установлен)
Скачай с https://git-scm.com/download/win и установи.

---

## Шаг 2 — Создать репозиторий на GitHub
1. Зайди на https://github.com
2. Нажми кнопку **New** (зелёная, справа вверху)
3. Название: `forecastiq` (или любое другое)
4. Видимость: **Public** (обязательно для бесплатного Streamlit Cloud)
5. Нажми **Create repository**

---

## Шаг 3 — Загрузить файлы на GitHub

Открой терминал (cmd или PowerShell) в папке с проектом:

```bash
# Инициализация
git init
git add .
git commit -m "Initial commit"

# Подключение к GitHub (замени USERNAME и REPO на свои)
git remote add origin https://github.com/USERNAME/forecastiq.git
git branch -M main
git push -u origin main
```

Или просто перетащи файлы через интерфейс GitHub (кнопка "uploading an existing file").

---

## Шаг 4 — Деплой на Streamlit Cloud
1. Зайди на https://share.streamlit.io
2. Войди через GitHub аккаунт
3. Нажми **New app**
4. Выбери:
   - Repository: `USERNAME/forecastiq`
   - Branch: `main`
   - Main file: `app.py`
5. Нажми **Deploy!**

⏱ Деплой займёт 2–5 минут.

После этого получишь постоянную ссылку вида:
`https://USERNAME-forecastiq-app-XXXXX.streamlit.app`

---

## Важно про большой датасет (200MB+)

Твой оригинальный файл слишком большой для GitHub (лимит 100MB).
Два варианта:

**Вариант A (рекомендую):** Загружать файл прямо через интерфейс сайта.
Сайт принимает файлы до 500MB через кнопку загрузки — это уже настроено.

**Вариант B:** Если хочешь чтобы данные загружались автоматически —
используй Google Drive или Dropbox + прямую ссылку (усложняет код).

---

## Проверка локально (опционально)

Если хочешь проверить до деплоя:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Откроется браузер на http://localhost:8501

---

## Готово! 🎉
