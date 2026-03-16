/**
 * KHNP Education AI Platform - Collaboration Portal
 * Interactive JS for tab navigation, notes, and local storage
 */

// ===== Tab Navigation =====
document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        // Remove active from all tabs and sections
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));

        // Activate clicked tab and corresponding section
        tab.classList.add('active');
        const sectionId = tab.dataset.tab;
        document.getElementById(sectionId).classList.add('active');
    });
});

// ===== Notes System (LocalStorage) =====
const STORAGE_KEY = 'khnp_edu_ai_notes';

function loadNotes() {
    const data = localStorage.getItem(STORAGE_KEY);
    return data ? JSON.parse(data) : { upstage: [], ey: [], general: [], decisions: [] };
}

function saveNotes(notes) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(notes));
}

function renderNotes(category) {
    const notes = loadNotes();
    const list = document.getElementById(`${category}-notes-list`);
    if (!list) return;

    list.innerHTML = notes[category].map((note, idx) => `
        <div style="padding: 14px; margin-bottom: 12px; background: var(--bg-gray); border-radius: 8px; border-left: 4px solid var(--khnp-primary);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.78rem; color: var(--text-secondary);">${note.date}</span>
                <button onclick="deleteNote('${category}', ${idx})" style="background: none; border: none; color: #dc2626; cursor: pointer; font-size: 0.8rem;">삭제</button>
            </div>
            <div style="font-size: 0.88rem; white-space: pre-wrap; line-height: 1.7;">${escapeHtml(note.content)}</div>
        </div>
    `).join('');
}

function addNote(category) {
    const input = document.getElementById(`${category}-note-input`);
    const content = input.value.trim();
    if (!content) return;

    const notes = loadNotes();
    notes[category].unshift({
        date: new Date().toLocaleString('ko-KR'),
        content: content
    });
    saveNotes(notes);
    renderNotes(category);
    input.value = '';
}

function deleteNote(category, index) {
    if (!confirm('이 기록을 삭제하시겠습니까?')) return;
    const notes = loadNotes();
    notes[category].splice(index, 1);
    saveNotes(notes);
    renderNotes(category);
}

function addDecision() {
    const input = document.getElementById('decision-input');
    const content = input.value.trim();
    if (!content) return;

    const notes = loadNotes();
    notes.decisions.unshift({
        date: new Date().toLocaleString('ko-KR'),
        content: content
    });
    saveNotes(notes);
    renderDecisions();
    input.value = '';
}

function renderDecisions() {
    const notes = loadNotes();
    const container = document.getElementById('decision-log');

    if (notes.decisions.length === 0) {
        container.innerHTML = '<div style="padding: 20px; text-align: center; color: var(--text-secondary); font-size: 0.9rem;">주요 의사결정 사항이 여기에 기록됩니다.</div>';
        return;
    }

    container.innerHTML = notes.decisions.map((d, idx) => `
        <div style="padding: 12px; margin-bottom: 8px; background: #fef3c7; border-radius: 8px; display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <div style="font-size: 0.75rem; color: #92400e; margin-bottom: 4px;">${d.date}</div>
                <div style="font-size: 0.88rem; line-height: 1.6;">${escapeHtml(d.content)}</div>
            </div>
            <button onclick="deleteDecision(${idx})" style="background: none; border: none; color: #dc2626; cursor: pointer; font-size: 0.8rem; flex-shrink: 0;">삭제</button>
        </div>
    `).join('');
}

function deleteDecision(index) {
    if (!confirm('이 결정사항을 삭제하시겠습니까?')) return;
    const notes = loadNotes();
    notes.decisions.splice(index, 1);
    saveNotes(notes);
    renderDecisions();
}

// ===== Export =====
function exportNotes(category) {
    const notes = loadNotes();
    const categoryNotes = notes[category];
    if (!categoryNotes || categoryNotes.length === 0) {
        alert('내보낼 기록이 없습니다.');
        return;
    }

    const categoryNames = { upstage: 'Upstage', ey: 'EY', general: '전체' };
    let text = `# KHNP Education AI Platform - ${categoryNames[category]} 회의/소통 기록\n`;
    text += `# 내보내기 일시: ${new Date().toLocaleString('ko-KR')}\n\n`;

    categoryNotes.forEach(note => {
        text += `---\n[${note.date}]\n${note.content}\n\n`;
    });

    downloadText(`khnp_notes_${category}_${Date.now()}.txt`, text);
}

function exportAllNotes() {
    const notes = loadNotes();
    let text = `# KHNP Education AI Platform - 전체 기록\n`;
    text += `# 내보내기 일시: ${new Date().toLocaleString('ko-KR')}\n\n`;

    const categories = [
        ['upstage', 'Upstage 소통 기록'],
        ['ey', 'EY 소통 기록'],
        ['general', '전체 회의록'],
        ['decisions', '의사결정 로그'],
    ];

    categories.forEach(([key, title]) => {
        text += `\n${'='.repeat(60)}\n## ${title}\n${'='.repeat(60)}\n\n`;
        (notes[key] || []).forEach(note => {
            text += `[${note.date}]\n${note.content}\n\n---\n\n`;
        });
    });

    downloadText(`khnp_all_notes_${Date.now()}.txt`, text);
}

function downloadText(filename, text) {
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// ===== SAT Detail =====
const satDetails = {
    analysis: {
        title: 'Analysis - 직무분석',
        subtitle: '교육 필요점 도출 및 학습목표 설정',
        content: `
            <table class="status-table">
                <thead><tr><th>기능</th><th>AI 역할</th><th>인간 역할</th></tr></thead>
                <tbody>
                    <tr><td>직무분석(JTA) 입력</td><td>JTA 문서 파싱 및 구조화</td><td>JTA 결과 검증</td></tr>
                    <tr><td>학습목표 매핑</td><td>RAG로 관련 규정/역량 자동 연계</td><td>학습목표 최종 확정</td></tr>
                    <tr><td>교육 필요점 분석</td><td>기존 교육과정과 JTA 갭 분석</td><td>우선순위 결정</td></tr>
                </tbody>
            </table>`
    },
    design: {
        title: 'Design - 교육설계',
        subtitle: '교육과정 구조 및 평가전략 설계',
        content: `
            <table class="status-table">
                <thead><tr><th>기능</th><th>AI 역할</th><th>인간 역할</th></tr></thead>
                <tbody>
                    <tr><td>교육과정 구조 설계</td><td>학습목표 기반 구조 자동 제안</td><td>구조 확정/수정</td></tr>
                    <tr><td>평가전략 수립</td><td>Bloom 수준별 문항 배분 제안</td><td>평가 기준 최종 결정</td></tr>
                    <tr><td>교수전략 선정</td><td>학습 유형별 적합 전략 추천</td><td>교수법 최종 선택</td></tr>
                </tbody>
            </table>`
    },
    development: {
        title: 'Development - 교재개발 (PoC 핵심 영역)',
        subtitle: 'AI 시스템이 가장 직접적으로 기여하는 SAT 단계',
        content: `
            <table class="status-table">
                <thead><tr><th>기능</th><th>AI 역할</th><th>인간 역할</th></tr></thead>
                <tbody>
                    <tr><td>강의 스크립트 작성</td><td>슬라이드 기반 초안 자동 생성</td><td>검토/수정, 경험 보충</td></tr>
                    <tr><td>평가문항 개발</td><td>학습목표 기반 문항 자동 생성</td><td>정답 검증, 난이도 조정</td></tr>
                    <tr><td>보충자료 연계</td><td>RAG로 관련 규정/사례 자동 검색</td><td>적절성 판단, 추가 자료 지정</td></tr>
                    <tr><td>시각자료 해석</td><td>VLM으로 도면/차트 설명 생성</td><td>기술적 정확성 확인</td></tr>
                </tbody>
            </table>`
    },
    implementation: {
        title: 'Implementation - 교육실행',
        subtitle: '교육 운영 및 강사 지원',
        content: `
            <table class="status-table">
                <thead><tr><th>기능</th><th>AI 역할</th><th>인간 역할</th></tr></thead>
                <tbody>
                    <tr><td>강사 가이드 제공</td><td>슬라이드별 스크립트, 시간 배분</td><td>교육 실행, 질의응답</td></tr>
                    <tr><td>학습자 맞춤화</td><td>수준별 보충설명 자동 제공</td><td>현장 분위기 조절</td></tr>
                    <tr><td>교육자료 갱신</td><td>최신 규정 변경 자동 감지/반영</td><td>변경사항 승인</td></tr>
                </tbody>
            </table>`
    },
    evaluation: {
        title: 'Evaluation - 평가환류',
        subtitle: '교육 효과성 분석 및 개선 환류',
        content: `
            <table class="status-table">
                <thead><tr><th>기능</th><th>AI 역할</th><th>인간 역할</th></tr></thead>
                <tbody>
                    <tr><td>문항 분석</td><td>정답률, 변별도, 난이도 자동 통계</td><td>문항 수정/교체 결정</td></tr>
                    <tr><td>교육효과 분석</td><td>학습목표 달성도 자동 분석</td><td>개선 방향 수립</td></tr>
                    <tr><td>피드백 환류</td><td>분석 결과 기반 교재 개선 제안</td><td>개선안 승인/적용</td></tr>
                    <tr><td>트렌드 분석</td><td>연도별 성취도 변화 추적</td><td>중장기 교육정책 반영</td></tr>
                </tbody>
            </table>`
    }
};

function showSATDetail(phase) {
    const detail = satDetails[phase];
    if (!detail) return;

    document.querySelectorAll('.sat-phase').forEach(p => p.classList.remove('active'));
    event.currentTarget.classList.add('active');

    document.getElementById('sat-detail-title').textContent = detail.title;
    document.getElementById('sat-detail-subtitle').textContent = detail.subtitle;
    document.getElementById('sat-detail-content').innerHTML = detail.content;
}

// ===== Utility =====
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== Initialize =====
document.addEventListener('DOMContentLoaded', () => {
    renderNotes('upstage');
    renderNotes('ey');
    renderNotes('general');
    renderDecisions();
});
