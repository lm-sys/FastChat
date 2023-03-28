let currentQuestionIndex = 0;

modelNameMapping = {
    "gpt35": "ChatGPT-3.5",
    "gpt4": "GPT-4",
    "alpaca": "Alpaca-13b",
    "vicuna": "Vicuna-13b",
    "llama": "LLaMA-13b",
    "bard": "Bard",
}

function formatText(input) {
    input = input.replace(/&/g, '&amp;')
                 .replace(/</g, '&lt;')
                 .replace(/>/g, '&gt;')
                 .replace(/"/g, '&quot;')
                 .replace(/'/g, '&#x27;')
                 .replace(/\//g, '&#x2F;');
    return input.replace(/\n/g, '<br>');
}

function text2Markdown(text) {
    // Normalize the text for markdown rendering.
    text = text.trim().replace('\n\n', '\n').replace('\n', '\n\n');
    return marked.parse(text);
}

function populateQuestions() {
    const select = document.getElementById('question-select');
    data.questions.forEach((question, index) => {
        const option = document.createElement('option');
        option.value = index;
        option.textContent = question.question;
        select.appendChild(option);
    });
}

function populateModels() {
    const select = document.getElementById('model-select');
    data.models.forEach(model => {
        const option = document.createElement('option');
        option.value = model;
        option.textContent = modelNameMapping[model];
        select.appendChild(option);
    });
}

function displayQuestion(index) {
    const question = data.questions[index].question;
    document.getElementById('selected-question').innerHTML = text2Markdown('**Question:** ' + question); // "<strong>Question: </strong>" + formatText(question);
    displayAnswers(index);
}

function displayAnswers(index) {
    const question = data.questions[index];
    const otherModel = document.getElementById('model-select').value;
    // document.getElementById('other-model-answer').innerHTML = formatText(question.answers[otherModel]);
    // document.getElementById('our-model-answer').innerHTML = formatText(question.answers.vicuna);
    // render the answers with markdown
    document.getElementById('other-model-answer').innerHTML = text2Markdown(question.answers[otherModel]);
    document.getElementById('our-model-answer').innerHTML = text2Markdown(question.answers.vicuna);

    // Display evaluation
    score = question.scores[otherModel];
    score_text = otherModel + " " + score[0] + "/10, Vicuna " + score[1] + "/10";
    document.getElementById('evaluation-header').textContent = "GPT-4 Evaluation" + " (Score: " + score_text + ")";
    // document.getElementById('evaluation-result').innerHTML = formatText(evaluationResult);
    document.getElementById('evaluation-result').innerHTML = text2Markdown(question.evaluations[otherModel]);

    // Update model names
    let assistant1_title = "Assistant #1"; // (" + modelNameMapping[otherModel] + ")";
    let assistant2_title = "Assistant #2 (Vicuna-13b, our model)";
    let assistant1_score_label = score[0].toString() + '/10';
    let assistant2_score_label = score[1].toString() + '/10';
    if (score[0] >= score[1]) {
        assistant1_title = '🏆 ' + assistant1_title;
        assistant1_score_label = '🏆 ' + assistant1_score_label;
    }
    if (score[0] <= score[1]) {
        assistant2_title = '🏆 ' + assistant2_title;
        assistant2_score_label = '🏆 ' + assistant2_score_label;
    }
    document.getElementById('other-model-header').textContent = assistant1_title;
    document.getElementById('our-model-header').textContent = assistant2_title;

    document.getElementById('other-score-label').textContent = assistant1_score_label;
    document.getElementById('our-score-label').textContent = assistant2_score_label;

    // Update expand buttons visibility for both cards after displaying answers
    // Reset the expanded state and update expand buttons visibility for both cards after displaying answers
    document.querySelectorAll('.expandable-card').forEach(card => {
        card.classList.remove('expanded');
        updateExpandButtonVisibility(card);
        const expandBtn = card.querySelector('.expand-btn');
        expandBtn.innerHTML = '<i class="material-icons" style="pointer-events: none">keyboard_arrow_down</i> Show more';   // .textContent = 'Show more';
    });
}

document.getElementById('question-select').addEventListener('change', e => {
    currentQuestionIndex = parseInt(e.target.value);
    displayQuestion(currentQuestionIndex);
});

// Update expand buttons whenever the model is changed
document.getElementById('model-select').addEventListener('change', () => {
    displayAnswers(currentQuestionIndex);
    document.querySelectorAll('.expandable-card').forEach(card => {
        updateExpandButtonVisibility(card);
    });
});

document.getElementById('prev-question').addEventListener('click', () => {
    currentQuestionIndex = Math.max(0, currentQuestionIndex - 1);
    document.getElementById('question-select').value = currentQuestionIndex;
    displayQuestion(currentQuestionIndex);
});

document.getElementById('next-question').addEventListener('click', () => {
    currentQuestionIndex = Math.min(data.questions.length - 1, currentQuestionIndex + 1);
    document.getElementById('question-select').value = currentQuestionIndex;
    displayQuestion(currentQuestionIndex);
});

function updateExpandButtonVisibility(card) {
    const cardTextContainer = card.querySelector('.card-text-container');
    const expandBtn = card.querySelector('.expand-btn');
    if (cardTextContainer.scrollHeight > cardTextContainer.offsetHeight) {
        expandBtn.style.display = 'flex';
    } else {
        expandBtn.style.display = 'none';
        card.classList.add('expanded');
    }
}

document.querySelectorAll('.expand-btn').forEach(btn => {
    btn.addEventListener('click', e => {
        const card = e.target.closest('.expandable-card');
        card.classList.toggle('expanded');
        const more = '<i class="material-icons" style="pointer-events: none">keyboard_arrow_down</i> Show more';
        const less = '<i class="material-icons" style="pointer-events: none">keyboard_arrow_up</i> Show less';
        e.target.innerHTML = card.classList.contains('expanded') ? less : more;
    });
});
