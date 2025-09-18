// const API_KEY = 'API_KEY'
export const BASE_URL = '/api';


export interface PipelineRequest {
  dataset: string;
  starting_time: string;
  variable: string;
}

export const fetchData = async (endpoint: string) => {
  const response = await fetch(`${BASE_URL}/${endpoint}`);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return await response.json();
};

export const postData = async (endpoint: string, body: PipelineRequest) => {
  const response = await fetch(`${BASE_URL}/${endpoint}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  return await response.json();
};