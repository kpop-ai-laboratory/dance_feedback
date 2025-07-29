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
    <div className="w-full max-w-7xl mx-auto px-6 py-12 relative">
      <h2 className="text-3xl font-bold mb-6">영상 업로드</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-12 mb-12 justify-items-center">
        <div className="w-full max-w-lg">
          <VideoUploadCard
            label="댄서 영상 업로드"
            file={dancer}
            onChange={file => setDancer(file)}
            inputId="dancer-input"
          />
        </div>
        <div className="w-full max-w-lg">
          <VideoUploadCard
            label="연습생 영상 업로드"
            file={trainee}
            onChange={file => setTrainee(file)}
            inputId="trainee-input"
          />
        </div>
      </div>

      <div className="flex justify-center mb-8">
        <button
          onClick={handleStartCompare}
          disabled={!(dancer && trainee) || isLoading}
          className="
            bg-red-300 hover:bg-red-400 text-white p-4 flex justify-center
            rounded-lg transition-colors duration-200 ease-in-out
            disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-red-200
          "
        >
          비교 시작
        </button>
      </div>

      <LoadingOverlay
        isLoading={isLoading}
        text="비교를 시작합니다. 최대 2시간이 소요될 수 있습니다."
      />
    </div>
  );
}
