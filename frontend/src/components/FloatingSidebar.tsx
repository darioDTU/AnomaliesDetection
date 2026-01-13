import type { Dispatch, SetStateAction } from "react";

type FloatingSidebarProps = {
    latitude: number;
    longitude: number;
    setLatitude: Dispatch<SetStateAction<number>>;
    setLongitude: Dispatch<SetStateAction<number>>;
}

const FloatingSidebar : React.FC<FloatingSidebarProps> = ({
    latitude,
    longitude,
    setLatitude,
    setLongitude
}) => (
    <div className="floating-sidebar">
        <div className="coord-adjust-section">
            <h3>Coordinates</h3>
            <div className="coord-input-group">
            <label>Latitude</label>
            <input type="number" value={latitude} onChange={e => setLatitude(Number(e.target.value))} />
            </div>
            <div className="coord-input-group">
            <label>Longitude</label>
            <input type="number" value={longitude} onChange={e => setLongitude(Number(e.target.value))} />
            </div>
        </div>
        </div>
);

export default FloatingSidebar;