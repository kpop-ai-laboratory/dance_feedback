import React, { useState, useEffect, useRef } from 'react';

const formatTime = sec => {
  const m = String(Math.floor(sec/60)).padStart(2,'0');
  const s = String(sec % 60).padStart(2,'0');
  return `${m}:${s}`;
};

export default function TestViewer() {
  const jobId = "79d449db2c2a431f870e2b520ff8835c";
  const [scores, setScores]         = useState([]);
  const [feedback, setFeedback]     = useState({});
  const [feedbackHistory, setFeedbackHistory] = useState([]);
  const videoRef = useRef();
  const threshold = 0.7;

  useEffect(() => {
    fetch(`/data/${jobId}/dancer_kp/scores.json`)
      .then(r=>r.json()).then(j=>setScores(j.frame_scores||[]));
    fetch(`/data/${jobId}/dancer_kp/feedback.json`)
      .then(r=>r.json()).then(j=>setFeedback(j));
  }, []);

  const onTimeUpdate = () => {
    if (!videoRef.current) return;
    // 1) 정확한 프레임 인덱스를 계산 (30fps 기준)
      const fps = 30;
      const frameIdx = Math.floor(videoRef.current.currentTime * fps);
    // 2) frame_scores, feedback 둘 다 "프레임 인덱스"를 키로 썼으니 frameIdx로 꺼내야 합니다
      const frameScore = scores[frameIdx];
      if (frameScore === undefined || frameScore >= threshold) return;
      
      const msgs = feedback[frameIdx];
      if (!msgs || msgs.length === 0) return;

    // t 계산

    // videoRef.current.currentTime 은 현재 비디오 재생 위치(초 단위, 소수 포함).
    
    // Math.floor(...) 로 소숫점 내림해서 “현재 몇 초”인지를 정수 t 로 구합니다.
    
    // 피드백 메시지 조회
    
    // feedback[t] 은 feedback.json 에서 키가 프레임 또는 초(여기서는 인덱스가 초) 인 배열 항목을 꺼내는 것.
    
    // !msgs || msgs.length===0 이면 “피드백 내용이 없거나” 이므로 바로 return; 해서 아무 처리도 안 하고 끝냅니다.
    
    // 유사도 비교
    
    // scores[t] 은 scores.json 의 frame_scores[t](혹은 second_scores[t]) 를 가리킵니다.
    
    // scores[t] >= threshold 이면 “유사도가 충분히 높아서” 굳이 피드백을 표시할 필요가 없으므로 return; 해서 건너뜁니다.


    // 3) “선생님 XX.X°” 과 “학생 YY.Y°” 추출해서
    //    둘 다 0° 이상인 메시지만 남기기
    const valid = msgs.filter(msg => {
      const tea = msg.match(/선생님\s*([\d.]+)°/);
      const stu = msg.match(/학생\s*([\d.]+)°/);
      if (!tea || !stu) return false;
      return parseFloat(tea[1]) > 0 && parseFloat(stu[1]) > 0;
    });
    if (valid.length === 0) return;

    // 4) 이미 기록된 시간인지 중복 체크
    const ts = formatTime(Math.floor(frameIdx / fps));
    if (feedbackHistory.some(e => e.time === ts)) return;

    // 5) 히스토리에 추가
    setFeedbackHistory(prev => [
      ...prev,
      { time: ts, msgs: valid }
    ]);
  };

  return (
    <div className="p-8 space-y-6">
      <h2 className="text-2xl font-bold">Test Viewer</h2>

      <video
        ref={videoRef}
        src={`/data/${jobId}/final_feedback_with_audio.mp4`}
        controls
        onTimeUpdate={onTimeUpdate}
        className="w-full max-w-2xl mx-auto border"
      />

      {feedbackHistory.length > 0 && (
        <div
          className="
            mt-6 p-4 bg-yellow-100 rounded shadow
            max-h-64 overflow-y-auto   /* ← 여기 */
            space-y-4
          "
        >
          <h3 className="font-semibold mb-2">피드백 히스토리</h3>
          {feedbackHistory.map((entry, idx) => (
            <div key={idx} className="mb-3">
              <div className="font-medium mb-1">⏱ {entry.time}</div>
              <ul className="list-disc list-inside text-sm">
                {entry.msgs.map((m,i) => <li key={i}>{m}</li>)}
              </ul>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
