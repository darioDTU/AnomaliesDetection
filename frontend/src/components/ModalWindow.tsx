import { ENDPOINTS } from '../service/endpoints.ts'
import { BASE_URL } from '../service/api.ts'
import { dbDescriptions, dbYears } from '../service/dbVariables.ts'
import '../css/Modal.css'

import type { Dispatch, SetStateAction } from 'react'

export type Statistics = Record<string, string>

type ModalWindowProps = {
  setHasPressButton: Dispatch<SetStateAction<boolean>>
  stats: Statistics
  selectedDb: string
}

const ModalWindow = ({ setHasPressButton, stats, selectedDb }: ModalWindowProps) => {
  const dbDescription = dbDescriptions[selectedDb] || 'Unknown Database'
  const years = dbYears[selectedDb] || []
  const startYear = years[0] || 'N/A'
  const endYear = years[years.length - 1] || 'N/A'

  return (
    <div className="modal-overlay" onClick={() => setHasPressButton(false)}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={() => setHasPressButton(false)}>
          ×
        </button>
        <div className="modal-header">
          <h2>Analysis Results</h2>
        </div>
        <div className="modal-body">
          <img
            src={`${BASE_URL}/${ENDPOINTS.showImage}?t=${Date.now()}`}
            alt="API result"
            className="modal-image"
          />
          <div className="modal-stats">
            <h4>Statistics</h4>
            <ul>
              {stats ? 
              Object.entries(stats).map(([key, value]) => (
              <li key={key}>
                  <strong>{key}:</strong> {value as string}
              </li>
              ))
              : <li>No statistics available for area with no anomalies</li>}
            </ul>
            <h4>Description</h4>
            <ul>
              <li>
                <strong>Database:</strong> {dbDescription} ({selectedDb})
              </li>
              <li>
                <strong>Climatology Period:</strong> {startYear} to {endYear}
              </li>
            </ul>
            <div className="modal-description">
              <p>
                This analysis uses data from the <a href="https://marine.copernicus.eu/" target="_blank" rel="noopener noreferrer">Copernicus Marine Environment Monitoring Service (CMEMS)</a>.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ModalWindow