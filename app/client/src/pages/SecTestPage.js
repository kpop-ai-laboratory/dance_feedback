import React, { useState, useEffect, useRef } from 'react';

const formatTime = sec => {
  const m = String(Math.floor(sec/60)).padStart(2,'0');
  const s = String(sec % 60).padStart(2,'0');
  return `${m}:${s}`;
};

export default function TestViewer() {
  const jobId = "75352c72579b47f2afea36edf2f23b4f";
  const [secScores, setSecScores] = useState([]);
  const [feedback, setFeedback]   = useState({});
  const [history, setHistory]     = useState([]);
  const videoRef = useRef();
  const threshold = 0.8;

  useEffect(() => {
    fetch(`/data/${jobId}/dancer_kp/scores.json`)
      .then(r => r.json())
      .then(j => {
        setSecScores(j.second_scores || []);
      });
    fetch(`/data/${jobId}/dancer_kp/feedback.json`)
      .then(r => r.json())
      .then(j => setFeedback(j));
  }, []);

  const onTimeUpdate = () => {
    if (!videoRef.current) return;
  
    const fps = 30;
    // 1) 현재 비디오 시간을 초 단위로
    const sec = videoRef.current.currentTime;
    // 2) 프레임 인덱스로 변환
    const frameIdx = Math.floor(sec * fps);
  
    // 3) 초단위 평균 유사도 판단 (옵셔널)
    const sim = secScores[Math.floor(sec)];
    if (sim === undefined || sim >= threshold) return;
  
    // 4) 프레임별 캡션 가져오기
    const msgs = feedback[frameIdx];
    if (!msgs || msgs.length === 0) return;

    // “선생님/학생 0°” 필터, 중복 체크 후 history 추가
    const valid = msgs.filter(m => {
      const tea = m.match(/선생님\s*([\d.]+)°/);
      const stu = m.match(/학생\s*([\d.]+)°/);
      return tea && stu && parseFloat(tea[1])>0 && parseFloat(stu[1])>0;
    });
    if (!valid.length) return;

    const ts = formatTime(Math.floor(sec)); 
    if (history.some(e => e.time === ts)) return;

    setHistory(h => [...h, { time: ts, msgs: valid }]);
  };

  return (
    <div className="p-8 space-y-6">
      <video
        ref={videoRef}
        src={`/data/${jobId}/final_feedback_with_audio.mp4`}
        controls
        onTimeUpdate={onTimeUpdate}
        className="w-full max-w-2xl mx-auto border"
      />

      {history.length > 0 && (
        <div className="mt-6 p-4 bg-yellow-100 rounded shadow max-h-64 overflow-y-auto">
          <h3 className="font-semibold mb-2">피드백 히스토리</h3>
          {history.map((e,i) => (
            <div key={i} className="mb-3">
              <div className="font-medium">⏱ {e.time}</div>
              <ul className="list-disc list-inside text-sm">
                {e.msgs.map((m,j) => <li key={j}>{m}</li>)}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
