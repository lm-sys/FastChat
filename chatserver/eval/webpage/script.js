let currentQuestionIndex = 0;

function formatText(input) {
    input = input.replace(/&/g, '&amp;')
                 .replace(/</g, '&lt;')
                 .replace(/>/g, '&gt;')
                 .replace(/"/g, '&quot;')
                 .replace(/'/g, '&#x27;')
                 .replace(/\//g, '&#x2F;');
    return input.replace(/\n/g, '<br>');
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
        option.textContent = model;
        select.appendChild(option);
    });
}

function displayQuestion(index) {
    const question = data.questions[index].question;

    document.getElementById('selected-question').innerHTML = "<strong>Question: </strong>" + formatText(question);
    displayAnswers(index);
}

function displayAnswers(index) {
    const question = data.questions[index];
    const otherModel = document.getElementById('model-select').value;
    document.getElementById('other-model-header').textContent = `AI Assistant 1 (${otherModel})`;
    let other_answer = question.answers[otherModel];
    let vicuna_answer = question.answers.vicuna;
    // Normalize the answers for markdown rendering.
    other_answer = other_answer.replace('\n\n', '\n').replace('\n', '\n\n');
    vicuna_answer = vicuna_answer.replace('\n\n', '\n').replace('\n', '\n\n');

    // document.getElementById('other-model-answer').innerHTML = formatText(question.answers[otherModel]);
    // document.getElementById('our-model-answer').innerHTML = formatText(question.answers.vicuna);
    // render the answers with markdown
    document.getElementById('other-model-answer').innerHTML = marked.parse(question.answers[otherModel]);
    document.getElementById('our-model-answer').innerHTML = marked.parse(question.answers.vicuna);

    displayEvaluation(index);
    // Update expand buttons visibility for both cards after displaying answers
    // Reset the expanded state and update expand buttons visibility for both cards after displaying answers
    document.querySelectorAll('.expandable-card').forEach(card => {
        card.classList.remove('expanded');
        updateExpandButtonVisibility(card);
        const expandBtn = card.querySelector('.expand-btn');
        expandBtn.textContent = 'Show more';
    });
}

function displayEvaluation(index) {
    const question = data.questions[index];
    const otherModel = document.getElementById('model-select').value;

    score = question.scores[otherModel];
    score_text = otherModel + " " + score[0] + "/10, Vicuna " + score[1] + "/10";
    document.getElementById('evaluation-header').textContent = "GPT-4 Evaluation" + " (Score: " + score_text + ")";

    evaluationResult = question.evaluations[otherModel];
    evaluationResult = evaluationResult.replace('\n\n', '\n').replace('\n', '\n\n');
    // document.getElementById('evaluation-result').innerHTML = formatText(evaluationResult);
    document.getElementById('evaluation-result').innerHTML = marked.parse(evaluationResult);
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
        expandBtn.style.display = 'block';
    } else {
        expandBtn.style.display = 'none';
    }
}

document.querySelectorAll('.expand-btn').forEach(btn => {
    btn.addEventListener('click', e => {
        const card = e.target.closest('.expandable-card');
        card.classList.toggle('expanded');
        e.target.textContent = card.classList.contains('expanded') ? 'Show less' : 'Show more';
    });
});
