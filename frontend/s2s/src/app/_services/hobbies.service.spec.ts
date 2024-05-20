import { TestBed } from '@angular/core/testing';

import { HobbyService } from './hobbies.service';

describe('HobbyService', () => {
  let service: HobbyService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(HobbyService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should create a list of 20 hobbies for preferences and scenarios', () => {
    expect(service.preferencesHobbies.length).toBe(20);
    expect(service.scenarioHobbies.length).toBe(20);
  });
});
