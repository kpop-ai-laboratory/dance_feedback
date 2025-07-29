import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import VideoUploadCard from '../components/VideoUploadCard';
import LoadingOverlay  from '../components/LoadingOverlay';

export default function Upload2() {
  const [dancer, setDancer]           = useState(null);
  const [trainee, setTrainee]         = useState(null);
  const [isLoading, setIsLoading]     = useState(false);

  const navigate = useNavigate();

  const handleStartCompare = async () => {
    if (!dancer || !trainee) return;
    setIsLoading(true);

    const form = new FormData();
    form.append('dancer', dancer);
    form.append('trainee', trainee);

    try {
      const res = await axios.post(
        '/compare/',
        form,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );

      // 서버에서 받은 결과 경로
      const finalVideo   = `/data/${res.data.final_video}`;
      const feedbackJson = `/data/${res.data.feedback_json}`;
      const scoresJson   = `/data/${res.data.scores_json}`;
      const durations    = res.data.durations;

      // 결과 페이지로 이동하며 state 전달
      navigate('/result', {
        state: { finalVideo, feedbackJson, scoresJson, durations }
      });
    } catch (err) {
      console.error(err);
      alert('서버 호출 중 에러가 발생했습니다');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex-grow flex items-center justify-center px-4">
        <div className="w-full max-w-5xl p-14 bg-white/15 backdrop-blur-md border border-white/30 rounded-2xl shadow-2xl">
          <h1 className="text-3xl md:text-4xl font-bold text-white mb-8 drop-shadow-md">
            영상 업로드
          </h1>
          
          <div className="grid md:grid-cols-2 gap-8 mb-8">
          <VideoUploadCard
            label="댄서 영상 업로드"
            file={dancer}
            onChange={file => setDancer(file)}
            inputId="dancer-input"
            className="bg-transparent text-white drop-shadow-md"
          />
        
          <VideoUploadCard
            label="연습생 영상 업로드"
            file={trainee}
            onChange={file => setTrainee(file)}
            inputId="trainee-input"
            className="bg-transparent text-white drop-shadow-md"
          />
        </div>
      

          {/* 두 개 버튼 */}
          <div className="flex flex-col md:flex-row gap-4 mb-6">
            <button
              onClick={handleStartCompare}
              disabled={!dancer || !trainee || isLoading}
              className="flex-1 py-4 bg-gradient-to-r from-pink-400 to-rose-500 hover:from-pink-500 hover:to-rose-600 text-white font-bold rounded-2xl shadow-lg disabled:opacity-50 transition drop-shadow-md"
            >
              {isLoading ? '비교 중…' : '동시 비교하기'}
            </button>
            <button
              // onClick={handleFeedback}
              disabled={!dancer || isLoading}
              className="flex-1 py-4 bg-gradient-to-r from-indigo-400 to-purple-500 hover:from-indigo-500 hover:to-purple-600 text-white font-bold rounded-2xl shadow-lg disabled:opacity-50 transition drop-shadow-md"
            >
              평가 영상 피드백
            </button>
          </div>
          
        </div>

      <LoadingOverlay
        isLoading={isLoading}
        text="비교를 시작합니다. 최대 2시간이 소요될 수 있습니다."
      />
  </main>
  );
}
