import './css/Dashboard.css'
import './css/Api.css'
import './css/Map.css'
import './css/Sidebar.css'
import { ENDPOINTS } from './service/endpoints.ts'
import { dbVariables, dbYears } from './service/dbVariables.ts'
import FloatingSidebar from './components/FloatingSidebar.tsx'
import Map from './components/Map.tsx'
import Sidebar from './components/Sidebar.tsx';
import { fetchData, postData, type PipelineRequest } from './service/api.ts'
import { useState, useEffect } from 'react'

import ErrorModalWindow from './components/ErrorModalWindow.tsx'
import ModalWindow from './components/ModalWindow.tsx'

function App() {
  const [depth, setDepth] = useState<number>(-1);
  const [latitude, setLatitude] = useState<number>(0);
  const [longitude, setLongitude] = useState<number>(0);
  const [selectedDb, setSelectedDb] = useState<string>('cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m');
  const [selectedVariable, setSelectedVariable] = useState<string>('thetao');
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('Classic');
  const [startTime, setStartTime] = useState<string>(
    (dbYears[selectedDb] && dbYears[selectedDb][0]) || ""
  );
  const dbList = ['cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m', 'cmems_mod_glo_phy_my_0.083deg_P1D-m', 'cmems_mod_glo_bgc_my_0.25deg_P1D-m'];
  const variableList = dbVariables[selectedDb] || [];
  const algorithmList = ['POT', 'Classic'];

  const [, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [stats, setStats] = useState<any>(null);

  const [hasPressButton, setHasPressButton] = useState<boolean>(false);
  const paramsReady = Boolean(
		latitude !== null && longitude !== null && selectedDb && startTime && selectedVariable
	)
   const requestBody: PipelineRequest = {
    dataset: selectedDb,
    latitude: latitude,
    longitude: longitude,
    starting_time: startTime,
    variable: selectedVariable,
    depth: depth
  };
  const handleApiCall = async () => {
    if (!paramsReady) return;
    setLoading(true);
    setError(null); 
    setHasPressButton(true);
    try {
      // const result = await fetchData(ENDPOINTS.run);
      // setData(result);
      let pipelineResult;
    if (selectedAlgorithm === 'POT') {
      pipelineResult = await postData(ENDPOINTS.runPOT, requestBody);
    } else {
      pipelineResult = await postData(ENDPOINTS.runClassic, requestBody);
    }
      setData(pipelineResult);
      // Fetch stats after running the pipeline
      const stats = await fetchData(ENDPOINTS.getStats);
      setStats(stats.Statistics);
    } catch (error) {
      console.error('Error fetching data:', error);
      let message = 'Unknown error';
      if (error && typeof error === 'object' && 'message' in error) {
        message = (error as { message: string }).message;
      } else if (typeof error === 'string') {
        message = error;
      }
      setError(`Failed to load: ${message}`);
    } finally {
      setLoading(false);
    }
  };

  const RECT_RESOLUTION: number = 2;
  const halfRes = RECT_RESOLUTION / 2;

  const rectangleBounds: [number, number][] = [
    [longitude - halfRes, latitude - halfRes], // SW
    [longitude + halfRes, latitude - halfRes], // SE
    [longitude + halfRes, latitude + halfRes], // NE
    [longitude - halfRes, latitude + halfRes], // NW
    [longitude - halfRes, latitude - halfRes], // Close polygon
  ];

  const handleMapClick = (lng : number, lat: number) => {
    setLongitude(lng);
    setLatitude(lat);
  };

  useEffect(() => {
    setStartTime((dbYears[selectedDb] && dbYears[selectedDb][0]) || "");
    const availableVars = dbVariables[selectedDb] || [];
    if (availableVars.length > 0 && !availableVars.includes(selectedVariable))
      setSelectedVariable(availableVars[0]);
  }, [selectedDb]);

  return (
    <div className="dashboard-bg">
      {/* <header className="dashboard-header">
        <div>
          <h1 className="dashboard-title">Anomalies Detection</h1>
          <p className="dashboard-subtitle">Advanced geospatial analytics dashboard</p>
        </div>
      </header> */}
      <div className="top-section">
        <aside className="sidebar">
          <Sidebar
            selectedDb={selectedDb}
            depth={depth}
            startTime={startTime}
            selectedVariable={selectedVariable}
            selectedAlgorithm={selectedAlgorithm}
            setSelectedDb={setSelectedDb}
            setStartTime={setStartTime}
            setSelectedVariable={setSelectedVariable}
            setSelectedAlgorithm={setSelectedAlgorithm}
            setDepth={setDepth}
            dbList={dbList}
            variableList={variableList}
            algorithmList={algorithmList}
            availableYears={dbYears[selectedDb] || []}
            
          />
        </aside>
        <div className="map-container">
          <button
            className="cute-btn run-btn-top"
            onClick={handleApiCall}
            disabled={!paramsReady || loading}
          >
            {loading ? 'Running...' : 'Run'}
          </button>
          <Map
              latitude={latitude}
              longitude={longitude}
              rectangleBounds={rectangleBounds}
              onMapClick={handleMapClick}
            />
          <FloatingSidebar
            latitude={latitude}
            longitude={longitude}
            setLatitude={setLatitude}
            setLongitude={setLongitude}
          />
        </div>
      </div>

      {!loading && hasPressButton && error == null && stats !== null && (
        <ModalWindow setHasPressButton={setHasPressButton} stats={stats} selectedDb={selectedDb} />
      )}

      {error && hasPressButton && (
        <ErrorModalWindow setHasPressButton={setHasPressButton} error={error} />
      )}
    </div>
  )
}

export default App
