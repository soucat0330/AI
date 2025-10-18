window.addEventListener("load", () => {
  const STORAGE_KEY = 'behavior_scheduler_db_v3';
  let db = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{"tasks":[],"records":[],"theme":"dark"}');

  const titleEl = document.getElementById('title');
  const deadlineEl = document.getElementById('deadline');
  const genreEl = document.getElementById('genre');
  const subjectiveEl = document.getElementById('subjective');
  const objectiveEl = document.getElementById('objective');
  const durationEl = document.getElementById('duration');
  const addBtn = document.getElementById('addTask');
  const generateBtn = document.getElementById('generate');
  const scheduleList = document.getElementById('scheduleList');
  const timelineRoot = document.getElementById('timeline');
  const feedbackTaskEl = document.getElementById('feedbackTask');
  const actualDurationEl = document.getElementById('actualDuration');
  const completedEl = document.getElementById('completed');
  const saveFeedbackBtn = document.getElementById('saveFeedback');
  const taskListEl = document.getElementById('taskList');
  const themeSelect = document.getElementById('themeSelect');
  const calendarEl = document.getElementById('calendar');

  function saveDB() { localStorage.setItem(STORAGE_KEY, JSON.stringify(db)); }

  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": "&#39;" })[c]); }

  function renderTasksInFeedback() {
    feedbackTaskEl.innerHTML = '';
    if (!db.tasks.length) {
      feedbackTaskEl.innerHTML = '<option value="">（タスクなし）</option>';
      return;
    }
    db.tasks.forEach((t, i) => {
      const opt = document.createElement('option');
      opt.value = i; opt.textContent = t.title;
      feedbackTaskEl.appendChild(opt);
    });
  }

  function renderTaskList() {
    taskListEl.innerHTML = '';
    if (!db.tasks.length) {
      taskListEl.innerHTML = '<div class="text-gray-400 text-sm">タスクがありません</div>';
      return;
    }
    db.tasks.forEach((t, i) => {
      const card = document.createElement('div');
      card.className = 'p-3 bg-gray-700 rounded flex justify-between items-center';
      const info = document.createElement('div');
      info.innerHTML = `<div class="font-semibold">${escapeHtml(t.title)}</div>
      <div class="text-sm text-gray-300">期限: ${t.deadline || '未設定'} / ジャンル: ${t.genre}</div>`;
      const del = document.createElement('button');
      del.textContent = '削除';
      del.className = 'bg-red-600 hover:bg-red-500 px-2 py-1 rounded text-sm';
      del.onclick = () => { db.tasks.splice(i, 1); saveDB(); renderTaskList(); renderTasksInFeedback(); renderCalendar(); renderLearnedPrioritiesFromModel(); }
      card.append(info, del);
      taskListEl.appendChild(card);
    });
  }

  addBtn.onclick = () => {
    const t = {
      title: titleEl.value.trim(),
      deadline: deadlineEl.value || '',
      genre: genreEl.value.trim() || '未設定',
      subjective: +subjectiveEl.value || 5,
      objective: +objectiveEl.value || 5,
      duration: +durationEl.value || 30
    };
    if(t.title.length > 20) return alert('タイトルが長すぎます(20文字以下にしてください)');
    if(t.duration < 0) return alert('見積時間エラー');
    if (!t.title) return alert('タイトルを入力');
    db.tasks.push(t);
    db.records.push({ task: t, completed: 0, actualDuration: null });
    saveDB();
    titleEl.value = deadlineEl.value = genreEl.value = subjectiveEl.value = objectiveEl.value = durationEl.value = '';
    renderTaskList(); renderTasksInFeedback(); renderCalendar(); renderLearnedPrioritiesFromModel();
  };

  function featurize(t) { return [(10 - t.subjective || 10) / 10, (10 - t.objective || 10) / 10, Math.log(1 + (t.duration || 30)) / Math.log(241)]; }
  let model = null;
  async function trainModel(records) {
    const data = records.filter(r => r.task);
    if (!data.length) return;
    const xs = tf.tensor2d(data.map(d => featurize(d.task)));
    const ys = tf.tensor2d(data.map(d => [d.completed ? 1 : 0]));
    model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [3], units: 8, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 4, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
    model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy' });
    await model.fit(xs, ys, { epochs: 15, batchSize: 8 });
    xs.dispose(); ys.dispose();
  }

  // --- 学習済み優先度表示 (下部) のためのユーティリティ関数 ---
  function getOrCreateLearnedEl() {
    let el = document.getElementById('learnedPriorities');
    if (!el) {
      el = document.createElement('div');
      el.id = 'learnedPriorities';
      el.className = 'mt-4 p-3 bg-gray-800 rounded';
      el.innerHTML = `<h3 class="font-semibold mb-2">学習済み優先度</h3><div class="learned-list"></div>`;
      // timeline の下に挿入できればそこへ、なければ body の最後へ
      if (timelineRoot && timelineRoot.parentNode) timelineRoot.parentNode.appendChild(el);
      else document.body.appendChild(el);
    }
    return el;
  }

  function renderLearnedPrioritiesFromScored(scored) {
    const el = getOrCreateLearnedEl();
    const list = el.querySelector('.learned-list');
    list.innerHTML = '';
    if (!scored || !scored.length) {
      list.innerHTML = '<div class="text-sm text-gray-400">表示する優先度がありません</div>';
      return;
    }
    scored.forEach(s => {
      const row = document.createElement('div');
      row.className = 'flex justify-between items-center text-sm py-1';
      const left = document.createElement('div');
      left.className = 'truncate'; left.textContent = s.title;
      const right = document.createElement('div');
      right.className = 'ml-2'; right.textContent = `${(s.score * 100).toFixed(1)}%`;
      row.appendChild(left); row.appendChild(right);
      list.appendChild(row);
    });
  }

  async function renderLearnedPrioritiesFromModel() {
    const el = getOrCreateLearnedEl();
    const list = el.querySelector('.learned-list');
    list.innerHTML = '';
    if (!db.tasks.length) {
      list.innerHTML = '<div class="text-sm text-gray-400">タスクがありません</div>';
      return;
    }
    if (!model) {
      list.innerHTML = '<div class="text-sm text-gray-400">モデルが学習されていません。生成ボタンを押すかフィードバックを保存してモデルを学習してください。</div>';
      return;
    }
    const xs = tf.tensor2d(db.tasks.map(t => featurize(t)));
    const preds = await model.predict(xs).data();
    xs.dispose();
    const arr = db.tasks.map((t, i) => ({ title: t.title, score: preds[i] })).sort((a, b) => b.score - a.score);
    arr.forEach(a => {
      const row = document.createElement('div');
      row.className = 'flex justify-between items-center text-sm py-1';
      const left = document.createElement('div'); left.className = 'truncate'; left.textContent = a.title;
      const right = document.createElement('div'); right.className = 'ml-2'; right.textContent = `${(a.score * 100).toFixed(1)}%`;
      row.appendChild(left); row.appendChild(right);
      list.appendChild(row);
    });
  }
  // --- /学習済み優先度表示 ---

  generateBtn.onclick = async () => {
    if (!db.tasks.length) return alert('タスクを追加してください');
    if (!model && db.records.length) await trainModel(db.records);
    const scored = await Promise.all(db.tasks.map(async t => {
      if (model) { const v = tf.tensor2d([featurize(t)]); const p = (await model.predict(v).data())[0]; v.dispose(); return { ...t, score: p }; }
      return { ...t, score: Math.random() * 0.5 + 0.25 };
    }));
    scored.sort((a, b) => b.score - a.score);
    scheduleList.innerHTML = '';
    scored.forEach(s => {
      const c = document.createElement('div'); c.className = 'p-3 bg-gray-700 rounded';
      c.innerHTML = `<div class="font-semibold">${s.title}</div>
      <div class="text-sm">優先度: ${(s.score * 100).toFixed(1)}%</div>`;
      scheduleList.appendChild(c);
    });
    renderTimeline(scored);
    renderCalendar();
    db.records.push(...scored.map(s => ({ task: s, completed: 0, actualDuration: null })));
    saveDB();
    // 生成したスコアを下部に表示
    renderLearnedPrioritiesFromScored(scored);
  };

  function renderTimeline(tasks) {
    timelineRoot.innerHTML = '';
    const total = tasks.reduce((a, b) => a + (b.duration || 30), 0);
    let cur = 0;
    tasks.forEach(t => {
      const r = document.createElement('div'); r.className = 'timeline-row';
      const l = document.createElement('div'); l.className = 'timeline-label'; l.textContent = t.title;
      const tr = document.createElement('div'); tr.className = 'timeline-track';
      const b = document.createElement('div'); b.className = 'timeline-bar';
      b.style.left = (cur / total) * 100 + '%'; b.style.width = ((t.duration || 30) / total) * 100 + '%';
      const hue = Math.round(200 - (t.score || 0) * 120);
      b.style.background = `hsl(${hue} 90% 65%)`; b.textContent = `${t.duration}m`;
      tr.appendChild(b); r.append(l, tr); timelineRoot.appendChild(r); cur += t.duration;
    });
  }

  // カレンダー描画
  function renderCalendar() {
    calendarEl.innerHTML = '';
    const now = new Date();
    const year = now.getFullYear(), month = now.getMonth();
    const first = new Date(year, month, 1);
    const last = new Date(year, month + 1, 0);
    const startDay = first.getDay(), days = last.getDate();
    for (let i = 0; i < startDay; i++) calendarEl.appendChild(document.createElement('div'));
    for (let d = 1; d <= days; d++) {
      const cell = document.createElement('div'); cell.className = 'calendar-cell';
      const dateStr = `${year}-${String(month + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
      cell.innerHTML = `<h4>${d}</h4>`;
      const todaysTasks = db.tasks.filter(t => t.deadline.startsWith(dateStr));
      todaysTasks.forEach(t => {
        const item = document.createElement('div');
        item.textContent = '• ' + t.title;
        item.className = 'text-xs truncate';
        cell.appendChild(item);
      });
      calendarEl.appendChild(cell);
    }
  }

  // テーマ切替
  function applyTheme(th) {
    const b = document.body;
    b.classList.remove('dark', 'light', 'blue', 'mono');
    if (th === 'light') { b.className = 'bg-gray-100 text-gray-900'; }
    else if (th === 'blue') { b.className = 'bg-blue-950 text-blue-100'; }
    else if (th === 'mono') { b.className = 'bg-gray-200 text-gray-800'; }
    else { b.className = 'bg-gray-900 text-gray-100'; }
    db.theme = th; saveDB();
  }
  themeSelect.onchange = () => applyTheme(themeSelect.value);

  saveFeedbackBtn.onclick = async () => {
    const idx = +feedbackTaskEl.value;
    if (isNaN(idx) || !db.tasks[idx]) return alert('選択してください');
    const dur = +actualDurationEl.value || db.tasks[idx].duration;
    const comp = +completedEl.value || 0;
    db.records.push({ task: db.tasks[idx], actualDuration: dur, completed: comp });
    await trainModel(db.records);
    saveDB(); alert('保存しました');
    // モデル学習後はモデルに基づく優先度表示を更新
    await renderLearnedPrioritiesFromModel();
  };

  applyTheme(db.theme);
  themeSelect.value = db.theme;
  renderTaskList(); renderTasksInFeedback(); renderCalendar();
  // 初期表示：もしモデルがあればモデルに基づく優先度を表示（通常は学習後に表示されます）
  renderLearnedPrioritiesFromModel();
});
