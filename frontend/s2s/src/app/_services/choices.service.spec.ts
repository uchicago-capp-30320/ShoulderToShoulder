import { TestBed } from '@angular/core/testing';

import { ChoicesService } from './choices.service';

describe('ChoicesService', () => {
  let service: ChoicesService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ChoicesService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
