'use client';

import { CalibrationDataPoint } from '@/types/calibration';
import { getSortedUniqueTargets } from '@/utils/targetOrdering';

interface ErrorDistributionProps {
  data: CalibrationDataPoint[];
}

export default function ErrorDistribution({ data }: ErrorDistributionProps) {
  if (data.length === 0) {
    return (
      <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
        <h2 className="text-xl font-bold mb-2 text-gray-800">Error distribution</h2>
        <p className="text-gray-600">No data available</p>
      </div>
    );
  }
  
  // Find max epoch safely
  const maxEpoch = data.reduce((max, item) => Math.max(max, item.epoch), 0);
  
  // Get latest epoch data for distribution analysis
  const latestData = data.filter(item => item.epoch === maxEpoch);
  
  const errorBins = [0, 0.05, 0.1, 0.20, 0.3, 0.5, 1.0, Infinity];
  
  const getErrorDistribution = () => {
    const distribution = errorBins.slice(0, -1).map((bin, i) => {
      const nextBin = errorBins[i + 1];
      const count = latestData.filter(item => 
        item.rel_abs_error >= bin && item.rel_abs_error < nextBin
      ).length;
      return {
        range: nextBin === Infinity ? `${(bin*100).toFixed(0)}%+` : `${(bin*100).toFixed(0)}-${(nextBin*100).toFixed(0)}%`,
        count,
        percentage: (count / latestData.length * 100).toFixed(1),
        color: bin < 0.05 ? 'bg-green-500' : bin < 0.20 ? 'bg-yellow-500' : 'bg-red-500'
      };
    });
    return distribution;
  };

  const distribution = getErrorDistribution();
  const maxCount = Math.max(...distribution.map(d => d.count));
  const targetNames = getSortedUniqueTargets(latestData);

  // Get top performing targets (lowest error)
  const topTargets = targetNames.map(targetName => {
    const targetData = latestData.filter(d => d.target_name === targetName);
    const avgError = targetData.reduce((sum, d) => sum + d.rel_abs_error, 0) / targetData.length;
    return { targetName, avgError };
  })
  .sort((a, b) => a.avgError - b.avgError) // Sort by error ascending (best first)
  .slice(0, 5);

  return (
    <div className="bg-white border border-gray-300 p-6 rounded-lg shadow-sm">
      <h2 className="text-xl font-bold mb-2 text-gray-800">Error distribution</h2>
      <p className="text-gray-600 mb-6">
        Distribution of relative absolute errors for epoch {maxEpoch}
      </p>

      <div className="space-y-3">
        {distribution.map((bin, i) => (
          <div key={i} className="flex items-center space-x-4">
            <div className="w-20 text-right text-sm font-mono text-gray-700">
              {bin.range}
            </div>
            <div className="flex-1 bg-gray-200 h-8 relative overflow-hidden rounded-lg">
              <div 
                className={`${bin.color} h-full transition-all duration-300`}
                style={{ width: `${maxCount > 0 ? (bin.count / maxCount) * 100 : 0}%` }}
              />
            </div>
            <div className="w-24 text-right text-sm font-mono text-gray-700">
              {bin.count} ({bin.percentage}%)
            </div>
          </div>
        ))}
      </div>

      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Quality targets</h3>
          <div className="space-y-2 text-sm">
            <div className="flex items-center">
              <div className="w-4 h-4 bg-green-500 rounded mr-2"></div>
              <span className="text-gray-600">Excellent: &lt;5% error</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 bg-yellow-500 rounded mr-2"></div>
              <span className="text-gray-600">Good: 5-20% error</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 bg-red-500 rounded mr-2"></div>
              <span className="text-gray-600">Needs work: &gt;20% error</span>
            </div>
          </div>
        </div>
        
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-3">Top calibration targets</h3>
          <div className="font-mono space-y-1 text-sm">
            {topTargets.map(({ targetName, avgError }) => (
              <div key={targetName} className="flex justify-between text-gray-600">
                <span className="truncate" title={targetName}>
                  {targetName.length > 20 ? targetName.substring(0, 20) + '...' : targetName}
                </span>
                <span>{avgError.toFixed(3)}</span>
              </div>
            ))}
            {targetNames.length > 5 && (
              <div className="text-xs text-gray-500 italic">
                +{targetNames.length - 5} more targets
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
