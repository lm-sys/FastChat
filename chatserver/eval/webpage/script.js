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
    const question = data.questions[index];
    document.getElementById('selected-question').textContent = question.question;
    displayAnswers(index);
}

function displayAnswers(index) {
    const question = data.questions[index];
    const otherModel = document.getElementById('model-select').value;
    document.getElementById('other-model-header').textContent = `AI Assistant 1 (${otherModel})`;
    document.getElementById('other-model-answer').innerHTML = formatText(question.answers[otherModel]);
    document.getElementById('our-model-answer').innerHTML = formatText(question.answers.vicuna);
    // document.getElementById('other-model-answer').innerHTML = marked.parse(question.answers[otherModel]);
    // document.getElementById('our-model-answer').innerHTML = marked.parse(question.answers.vicuna);
    displayEvaluation(index);
}

function displayEvaluation(index) {
    const question = data.questions[index];
    const otherModel = document.getElementById('model-select').value;
    // Here set innerHTML to the evaluation result directly, but it will be escaped.
    evaluationResult = question.evaluations[otherModel];
    document.getElementById('evaluation-result').innerHTML = formatText(evaluationResult);
}

document.getElementById('question-select').addEventListener('change', e => {
    currentQuestionIndex = parseInt(e.target.value);
    displayQuestion(currentQuestionIndex);
});

document.getElementById('model-select').addEventListener('change', () => {
    displayAnswers(currentQuestionIndex);
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
