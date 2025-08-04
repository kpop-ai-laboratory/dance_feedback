import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

export default function MediaUpload({ onUpload }) {
  const [previews, setPreviews] = useState([]);

  const onDrop = useCallback(async files => {
    const uploaded = [];
    for (let file of files) {
      const form = new FormData();
      form.append('file', file);
      const res = await axios.post('/board/upload_media', form, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      uploaded.push(res.data.url);
    }
    // 로컬 미리보기 + 실제 업로드 URL 함께 반환
    setPreviews(files.map((f,i)=>({
      url: URL.createObjectURL(f),
      uploadedUrl: uploaded[i],
      type: f.type
    })));
    onUpload(uploaded);
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': [], 'video/*': [] }
  });

  return (
    <div>
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed p-6 text-center
          cursor-pointer ${isDragActive?'border-green-400':'border-gray-300'}
        `}
      >
        <input {...getInputProps()} />
        {isDragActive 
          ? <p>드래그 중...</p> 
          : <p>이미지 드래그 또는 클릭하여 업로드</p>}
      </div>
      <div className="mt-4 grid grid-cols-3 gap-4">
        {previews.map((p,i)=>(
          <div key={i}>
            {p.type.startsWith('video') 
              ? <video src={p.url} controls className="w-full h-32 object-cover" /> 
              : <img src={p.url} className="w-full h-32 object-cover" alt="" />}
          </div>
        ))}
      </div>
    </div>
  );
}
