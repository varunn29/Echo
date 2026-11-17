import { useState } from 'react';

import Sidebar, { type Section } from './components/Sidebar';
import WelcomeSection from './components/WelcomeSection';
import WritingComposer from './components/WritingComposer';
import WritingData from './components/WritingData';
import TrainingStatusCard from './components/TrainingStatusCard';
import WritingStyleView from './components/WritingStyleView';
import SettingsView from './components/SettingsView';

import heroBackground from '../../assets/hero-background.png';

function App() {
  const [activeSection, setActiveSection] = useState<Section>('Dashboard');

  function renderContent() {
    if (activeSection === 'Dashboard') {
      return (
        <>
          <WelcomeSection />
          <div className="px-12 pb-6">
            <TrainingStatusCard />
          </div>
          <WritingComposer />
        </>
      );
    }

    if (activeSection === 'My Writing Data') {
      return (
        <WritingData
          initialTab="messages"
          onTabChange={(tab) =>
            setActiveSection(tab === 'messages' ? 'My Writing Data' : 'Personal Knowledge')
          }
        />
      );
    }

    if (activeSection === 'Personal Knowledge') {
      return (
        <WritingData
          initialTab="knowledge"
          onTabChange={(tab) =>
            setActiveSection(tab === 'messages' ? 'My Writing Data' : 'Personal Knowledge')
          }
        />
      );
    }

    if (activeSection === 'My Writing Style') {
      return <WritingStyleView onNavigate={setActiveSection} />;
    }

    if (activeSection === 'Settings') {
      return <SettingsView onNavigate={setActiveSection} />;
    }

    return null;
  }

  return (
    <div className="flex h-dvh w-full overflow-hidden bg-[#f8f6ef]">
      <Sidebar
        activeSection={activeSection}
        onNavigate={setActiveSection}
      />

      <main
        className="relative flex min-w-0 flex-1 flex-col overflow-y-auto"
        style={{
          backgroundImage: `url(${heroBackground})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
        }}
      >
        <div
          className="pointer-events-none fixed inset-y-0 right-0 z-0 bg-[#f8f6ef]/75"
          style={{ left: '224px' }}
        />

        <div className="relative z-10 flex min-h-full flex-1 flex-col">
          {renderContent()}
        </div>
      </main>
    </div>
  );
}

export default App;
